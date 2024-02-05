import torch
from PIL import Image
from models import *
from torchvision.transforms import Resize, CenterCrop, Compose, ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = Compose([
                        ToTensor(),
                        Resize(256, antialias=True),
                        CenterCrop(224)
                        ])

@torch.no_grad()
def caption_image(encoder, decoder, image_path, vocab, beam_size=3):
    # Criação de legenda de imagem à partir de método Beam Search.
    global img_transform
    encoder.eval()
    decoder.eval()

    image = Image.open(image_path)
    image = img_transform(image).to(device)

    encoded_image = encoder(image.unsqueeze(0))
    encoded_image = encoded_image.view(*encoded_image.size()[:2], -1)
    
    best_candidates = []
    possible_candidates = []
    final_candidates = []

    bos_token = torch.tensor([vocab["<bos>"]]).to(device)
    hidden, cell = decoder._init_hidden_and_cell_state(1)
    top_k_scores, top_k_tokens, hidden, cell = get_top_k_predictions(decoder, encoded_image, bos_token, hidden, cell, beam_size)
    top_k_scores = top_k_scores.squeeze(0)
    top_k_tokens = top_k_tokens.squeeze(0)
    
    best_candidates.extend((top_k_score, [top_k_token], hidden, cell) for top_k_score, top_k_token in zip(top_k_scores, top_k_tokens))

    it = 1
    while beam_size > 0:
        possible_candidates = []
        for candidate in best_candidates:
            score, tokens, hidden, cell = candidate
            top_k_scores, top_k_tokens, hidden, cell = get_top_k_predictions(decoder, encoded_image, tokens[-1].unsqueeze(0), hidden, cell, beam_size)
            top_k_scores = top_k_scores.squeeze(0)
            top_k_tokens = top_k_tokens.squeeze(0)

            possible_candidates.extend((score + top_k_score, tokens + [top_k_token], hidden, cell) for top_k_score, top_k_token in zip(top_k_scores, top_k_tokens))
        best_candidates = sorted(possible_candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        for candidate in best_candidates:
            if candidate[1][-1] == vocab["<eos>"]:
                final_candidates.append(candidate)
                best_candidates.remove(candidate)
                beam_size -= 1

    caption = sorted(final_candidates, key=lambda x: x[0], reverse=True)[0][1][:-1]

    return caption

@torch.no_grad()
def get_top_k_predictions(decoder, encoded_image, token, hidden, cell, k):
    context, _ = decoder.attention(encoded_image, hidden)

    embeded_caption = decoder.embedding(token)
    hidden, cell = decoder.lstm(torch.cat((context, embeded_caption), dim=1), (hidden, cell))
    preds = decoder.lstm_fc(hidden) # (1, vocab_size)

    top_k_scores, top_k_tokens = torch.topk(preds, k)
    return top_k_scores, top_k_tokens, hidden, cell

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
    checkpoint = torch.load("checkpoints/last.pth")
    vocab = torch.load("checkpoints/vocab.pth")
    encoder = Encoder().to(device)
    decoder = Decoder(hidden_size, len(vocab), emb_size, attention_size, img_emb_size, num_channels, dropout).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    image_path = "coco/images/val2017/000000579635.jpg"
    beam_size = 3
    caption = caption_image(encoder, decoder, image_path, vocab, beam_size)
    caption = [cap.item() for cap in caption]
    print(vocab.lookup_tokens(caption))

if __name__ == "__main__":
    main()
