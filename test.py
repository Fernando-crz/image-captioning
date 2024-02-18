import torch
from torchvision.datasets import CocoCaptions
from torchtext.data.utils import get_tokenizer
from datasets import load_metric
from dataset import *
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

    meteor = load_metric("meteor", trust_remote_code=True)
    bleu = load_metric("bleu", trust_remote_code=True)

    bleu_score = bleu.compute(predictions=hypotheses, references=references)
    meteor_score = meteor.compute(predictions=hypotheses, references=references)

    scores = {
        "BLEU_1": bleu_score["precisions"][0],
        "BLEU_2": bleu_score["precisions"][1],
        "BLEU_3": bleu_score["precisions"][2],
        "BLEU_4": bleu_score["precisions"][3],
        "METEOR": meteor_score["meteor"]
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
    torch.save(scores, "./checkpoints/scores.pth")

if __name__ == "__main__":
    main()