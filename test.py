import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import CocoCaptions
from torchtext.data.utils import get_tokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from time import time
from datasets import *
from models import *
from utils import *
from evaluation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_freq = 200

@torch.no_grad()
def test_model(coco_test_ds, encoder, decoder, vocab, beam_size=3):
    global print_freq, img_transform
    encoder.eval()
    decoder.eval()

    losses = []

    references = []
    hypotheses = []
    for i, (image, captions) in enumerate(coco_test_ds):
        generated_caption, _ = caption_image(encoder, decoder, image, vocab, beam_size=beam_size)

        hypotheses.append(generated_caption)
        references.append(captions)

        if i % print_freq == 0:
            print(f"[{i}] CAPTION GENERATED: {generated_caption}")
            print(f"FIRST REFERENCE: {captions[0]}")
            print(f"Percentage of progress: {100 * i / len(coco_test_ds)}%")
    
    weights = [
        (1./2., 1./2.),                 # BLEU_2 SCORE
        (1./3., 1./3., 1./3.),          # BLEU_3_SCORE
        (1./4., 1./4., 1./4., 1./4.)    # BLEU_4_SCORE
    ]

    bleu_score = corpus_bleu(references, hypotheses, weights=weights)
    meteor_scores = [meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    meteor_final_score = sum(meteor_scores) / len(meteor_scores)
    
    scores = {
        "BLEU_2_SCORE": bleu_score[0],
        "BLEU_3_SCORE": bleu_score[1],
        "BLEU_4_SCORE": bleu_score[2],
        "METEOR_SCORE": meteor_final_score
    }

    print(f"[DONE] SCORES FOUND: {scores}")
    return scores

def main():
    print("device found: ", device)
    # Parâmetros do decoder
    hidden_size = 512
    emb_size = 512
    attention_size = 512
    img_emb_size = 14 * 14
    num_channels = 512
    dropout = 0.4
    
    images_path = "./coco/images/val2017"
    annotations_path = "./coco/annotations/captions_val2017.json"
    vocab = torch.load("./checkpoints/vocab.pth")
    
    tokenizer = get_tokenizer("basic_english")
    coco_test_ds = CocoCaptions(
        root=images_path,
        annFile=annotations_path,
        target_transform=lambda str_list: [tokenizer(_str) for _str in str_list] 
    )
    # Carregando modelos
    checkpoint = torch.load("checkpoints/best.pth")
    encoder = Encoder().to(device)
    decoder = Decoder(hidden_size, len(vocab), emb_size, attention_size, img_emb_size, num_channels, dropout).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    scores = test_model(coco_test_ds, encoder, decoder, vocab)
    torch.save(scores, "/checkpoints/scores.pth")

if __name__ == "__main__":
    main()