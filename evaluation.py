import torch
from PIL import Image
from models import *
from torchvision.transforms import Resize, CenterCrop, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
from matplotlib import pyplot as plt
from math import ceil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = Compose([
                        ToTensor(),
                        Resize(256, antialias=True),
                        CenterCrop(224)
                        ])

@torch.no_grad()
def caption_image(encoder, decoder, image, vocab, beam_size=3):
    # Criação de legenda de imagem à partir de método Beam Search.
    global img_transform
    encoder.eval()
    decoder.eval()

    image = img_transform(image).to(device)

    encoded_image = encoder(image.unsqueeze(0))
    encoded_image = encoded_image.view(*encoded_image.size()[:2], -1)
    
    best_candidates = []
    possible_candidates = []
    final_candidates = []

    bos_token = torch.tensor([vocab["<bos>"]]).to(device)
    hidden, cell = decoder._init_hidden_and_cell_state(1)
    top_k_scores, top_k_tokens, hidden, cell, attention = get_top_k_predictions(decoder, encoded_image, bos_token, hidden, cell, beam_size)
    top_k_scores = top_k_scores.squeeze(0)
    top_k_tokens = top_k_tokens.squeeze(0)
    
    best_candidates.extend((top_k_score, [top_k_token], hidden, cell, [attention]) for top_k_score, top_k_token in zip(top_k_scores, top_k_tokens))

    while beam_size > 0:
        possible_candidates = []
        for candidate in best_candidates:
            score, tokens, hidden, cell, attention_list = candidate
            top_k_scores, top_k_tokens, hidden, cell, attention = get_top_k_predictions(decoder, encoded_image, tokens[-1].unsqueeze(0), hidden, cell, beam_size)
            top_k_scores = top_k_scores.squeeze(0)
            top_k_tokens = top_k_tokens.squeeze(0)

            possible_candidates.extend((score + top_k_score, tokens + [top_k_token], hidden, cell, attention_list + [attention]) for top_k_score, top_k_token in zip(top_k_scores, top_k_tokens))
        best_candidates = sorted(possible_candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        candidates_to_keep = []
        for candidate in best_candidates:
            if candidate[1][-1] == vocab["<eos>"]:
                final_candidates.append(candidate)
                beam_size -= 1
            else:
                candidates_to_keep.append(candidate)
        
        best_candidates = candidates_to_keep

    chosen_candidate = sorted(final_candidates, key=lambda x: x[0], reverse=True)[0]
    caption = chosen_candidate[1][:-1]
    attention_list = chosen_candidate[4][:-1]

    caption = [cap.item() for cap in caption]
    caption = vocab.lookup_tokens(caption)

    return caption, attention_list

def string_from_caption(caption):
    return " ".join(caption)

@torch.no_grad()
def get_top_k_predictions(decoder, encoded_image, token, hidden, cell, k):
    context, attention = decoder.attention(encoded_image, hidden)
    embeded_caption = decoder.embedding(token)
    hidden, cell = decoder.lstm(torch.cat((context, embeded_caption), dim=1), (hidden, cell))
    preds = decoder.lstm_fc(hidden) # (1, vocab_size)

    top_k_scores, top_k_tokens = torch.topk(preds, k)
    return top_k_scores, top_k_tokens, hidden, cell, attention

def visualize_attention(encoder, decoder, image, vocab, beam_size=3):
    caption, attention_list = caption_image(encoder, decoder, image, vocab, beam_size)

    upsampler = torch.nn.Upsample(scale_factor=16, mode="bilinear")
    attention_list = [upsampler(attention.view((1, 1, 14, 14))) for attention in attention_list]
    attention_list = [to_pil_image(attention.squeeze(0)) for attention in attention_list]

    image = img_transform(image).to(device)
    image = to_pil_image(image)

    plot_figure = plt.figure(figsize=(20, 12))
    plt.subplot(ceil((len(caption) + 1) / 5), 5, 1)
    plt.imshow(image)
    plt.set_cmap(cm.Greys_r)
    plt.axis("off")
    
    for i, (word, attention) in enumerate(zip(caption, attention_list)):
        if i >= 50:
            break
        plt.subplot(ceil((len(caption) + 1) / 5), 5, i + 2)

        plt.text(0, 1, str(word), color="black", backgroundcolor="white", fontsize=12)
        plt.imshow(image)
        plt.imshow(attention, alpha=0.7)
        plt.set_cmap(cm.Greys_r)
        plt.axis("off")
    
    plot_figure.canvas.draw()
    plot_image = Image.frombytes("RGB", plot_figure.canvas.get_width_height(), plot_figure.canvas.tostring_rgb())
    plt.close()
    return plot_image

def main():
    print("device found: ", device)
    # Parâmetros do decoder
    hidden_size = 512
    emb_size = 512
    attention_size = 512
    img_emb_size = 14 * 14
    num_channels = 512
    dropout = 0.4

    # Carregando modelos
    checkpoint = torch.load("checkpoints/best.pth")
    vocab = torch.load("checkpoints/vocab.pth")
    encoder = Encoder().to(device)
    decoder = Decoder(hidden_size, len(vocab), emb_size, attention_size, img_emb_size, num_channels, dropout).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    image_path = r"images\20240214_214626.jpg"
    image = Image.open(image_path)
    
    beam_size = 3
    caption, _ = caption_image(encoder, decoder, image, vocab, beam_size)
    plot_image = visualize_attention(encoder, decoder, image, vocab, beam_size)
    plot_image.show()

if __name__ == "__main__":
    main()
