import torch
import torch.nn.functional as F
import argparse
import pickle
import os
from torchvision import transforms
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from custom_datasets import BrunelloImageDataModule, collate_fn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def main():
    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",)
    data_module = BrunelloImageDataModule(
        tokenizer=tokenizer, batch_size=1, num_workers=1, collate=collate_fn,
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    checkpoint = "BEST_checkpoint_brunello_loadgpt.pth.tar"  # model checkpoint
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # sets device for model and PyTorch tensors
    # cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Load model
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint["encoder"]
    encoder = encoder.to(device)
    encoder.eval()

    vocab_size = 50257
    beam_size = 4
    pad_token = 50256

    # # Prepare an image
    # image = load_image(args.image, transform)
    # image_tensor = image.to(device)

    for i, (image, captions, lengths) in enumerate(test_loader):
        # images = images.to(device)
        # # Generate an caption from the image
        # feature = encoder(images)
        # sampled_ids = decoder.sample(feature)
        # sampled_ids = (
        #     sampled_ids[0].cpu().numpy()
        # )  # (1, max_seq_length) -> (max_seq_length)
        # sentence = tokenizer.decode(sampled_ids)
        # print(sentence)
        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(
            1, -1, encoder_dim
        )  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(
            k, num_pixels, encoder_dim
        )  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[pad_token]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(
                encoder_out, h
            )  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c)
            )  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(
                    k, 0, True, True
                )  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
            )  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != pad_token
            ]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 200:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        print(seq)
        break


if __name__ == "__main__":
    main()
