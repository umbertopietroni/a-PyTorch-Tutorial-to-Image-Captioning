import time
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention

# from datasets import *
from utils import *

# from nltk.translate.bleu_score import corpus_bleu

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from custom_datasets import BrunelloImageDataModule, collate_fn, HMImageDataModule

# # Data parameters
# data_folder = (
#     "/media/ssd/caption data"  # folder with data files saved by create_input_files.py
# )


# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 1  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 8
workers = 4  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.0  # clip gradients at an absolute value of
alpha_c = (
    1.0  # regularization parameter for 'doubly stochastic attention', as in the paper
)
best_bleu4 = 0.0  # BLEU-4 score right now
best_val_loss = float("inf")
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
load_gpt = True
brunello = False


def main():
    """
    Training and validation.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, word_map, best_val_loss, load_gpt, brunello
    dataset_name = "brun" if brunello else "hm"
    data_name = "{}_loadgpt".format(dataset_name)  # base name shared by data files
    print(data_name)
    # base_path = "data/hm"

    # Load vocabulary wrapper
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",)
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2embedding = gpt2.transformer.wte
    vocab_len, embed_size = (
        gpt2embedding.num_embeddings,
        gpt2embedding.embedding_dim,
    )
    hidden_size = embed_size
    lm_head = gpt2.lm_head

    emb_dim = embed_size  # dimension of word embeddings
    attention_dim = embed_size  # dimension of attention linear layers
    decoder_dim = embed_size  # dimension of decoder RNN

    if brunello:
        data_module = BrunelloImageDataModule(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=workers,
            collate=collate_fn,
        )
    else:
        data_module = HMImageDataModule(
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=workers,
            collate=collate_fn,
        )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_len,
            dropout=dropout,
        )
        if load_gpt:
            decoder.embedding = gpt2embedding
            decoder.fc = lm_head
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder
            else None
        )

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_improvement = checkpoint["epochs_since_improvement"]
        best_val_loss = checkpoint["bleu-4"]
        decoder = checkpoint["decoder"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        encoder = checkpoint["encoder"]
        encoder_optimizer = checkpoint["encoder_optimizer"]
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 5:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
        )

        # One epoch's validation
        recent_val_loss = validate(
            val_loader=val_loader, encoder=encoder, decoder=decoder, criterion=criterion
        )

        # Check if there was an improvement
        # is_best = recent_bleu4 > best_bleu4
        # best_bleu4 = max(recent_bleu4, best_bleu4)
        is_best = recent_val_loss < best_val_loss
        best_val_loss = min(recent_val_loss, best_val_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(
            data_name,
            epoch,
            epochs_since_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            recent_val_loss,
            is_best,
        )


def train(
    train_loader,
    encoder,
    decoder,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens
        )

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top5=top5accs,
                )
            )


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print(
                    "Validation: [{0}/{1}]\t"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top5=top5accs,
                    )
                )

        print(
            "\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}".format(
                loss=losses, top5=top5accs
            )
        )

    return losses.avg


if __name__ == "__main__":
    main()
