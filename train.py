import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from time import time
from datasets import *
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
print(f"DEVICE DEFINED: {device}")

# Parâmetros de modelos
hidden_size = 512
emb_size = 512
attention_size = 512
img_emb_size = 14 * 14
num_channels = 512
dropout = 0.4

# Parâmetros de treinamento
print_freq = 200
email_freq = 9
encoder_lr = 1e-4
decoder_lr = 4e-4
start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 110
grad_clip = 5.
best_score = 0.
fine_tune_encoder = False
load_checkpoint = False
checkpoint_path = "./checkpoints/best.pth"

def train(train_dl, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, pad):
    if fine_tune_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()

    losses = []

    for i, (images, captions, captions_len) in enumerate(train_dl):
        images = images.to(device)
        captions = captions.to(device)
        
        encoded_images = encoder(images)

        encoded_images = encoded_images.view(*encoded_images.size()[:2], -1)

        prediction, attention_weights, sorted_indices = decoder(encoded_images, captions, captions_len - 1)
        
        # Comparando predições à partir do token após <bos>, por isso o captions[:, 1:]
        prediction = prediction[sorted_indices]
        captions = captions[:, 1:][sorted_indices]
        sorted_indices = sorted_indices.to(captions_len.device)
        captions_len = captions_len[sorted_indices] - 1

        packed_prediction = pack_padded_sequence(prediction, captions_len , batch_first=True).data
        packed_captions = pack_padded_sequence(captions, captions_len , batch_first=True).data

        loss = criterion(packed_prediction, packed_captions)
        loss += (1. - attention_weights.sum(dim=1)**2).mean()

        losses.append(loss.item())

        if fine_tune_encoder:
            encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        if fine_tune_encoder:
            nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        
        if fine_tune_encoder:
            encoder_optimizer.step()
        decoder_optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Percentage Done: {(i / len(train_dl) * 100):.2f}% Loss: {loss:.4f}\n")

    return losses

def validate(coco_val_ds, encoder, decoder, criterion, pad):
    encoder.eval()
    decoder.eval()

    losses = []

    references = []
    hypotheses = []

    with torch.no_grad():
        for i, (images, captions, captions_len) in enumerate(coco_val_ds):
            images = images.to(device)
            captions = captions.to(device)
            
            encoded_images = encoder(images)

            encoded_images = encoded_images.view(*encoded_images.size()[:2], -1)
            
            prediction, attention_weights, sorted_indices = decoder(encoded_images, captions, captions_len - 1)
        
            # Comparando predições à partir do token após <bos>, por isso o captions[:, 1:]
            prediction = prediction[sorted_indices]
            captions = captions[:, 1:][sorted_indices]
            sorted_indices = sorted_indices.to(captions_len.device)
            captions_len = captions_len[sorted_indices] - 1

            packed_prediction = pack_padded_sequence(prediction, captions_len , batch_first=True).data
            packed_captions = pack_padded_sequence(captions, captions_len , batch_first=True).data

            loss = criterion(packed_prediction, packed_captions)
            loss += (1. - attention_weights.sum(dim=1)**2).mean()

            losses.append(loss.item())

            if i % print_freq == 0:
                print(f"Validation Batch: {i}, Percentage done: {(i / len(coco_val_ds) * 100):.2f}%, Loss: {loss:.4f}\n")

            # Aqui, vamos gerar o conjunto de referências e hipóteses para o cálculo da score BLEU.
            # Referências:
            caption_list = []
            for caption, caption_len in zip(captions, captions_len):
                caption_list.append(caption[:(caption_len.item() + 1)].tolist())
            
            # O batch inteiro está se referindo à mesma imagem. Tendo isso em mente, cada hipótese vai
            # se referir ao mesmo conjunto de captações presentes no conjunto de validação. Logo,
            # Vamos adicionar caption_list na mesma quantidade que temos batches.
            for _ in range(prediction.size(0)):
                references.append(caption_list)

            hypothesis = prediction.argmax(dim=2).tolist()
            hypotheses.extend(hypothesis)
    
    bleu_score = corpus_bleu(references, hypotheses)

    print(f"[DONE] BLEU SCORE FOUND: {bleu_score}")

    return bleu_score, losses

def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print(f"New learning rate is {optimizer.param_groups[0]['lr']}\n")

def main():
    global best_score, epochs_since_improvement, start_epoch
    images_path_train = "./coco/images/train2017"
    annotations_path_train = "./coco/annotations/captions_train2017.json"
    images_path_val = "./coco/images/val2017"
    annotations_path_val = "./coco/annotations/captions_val2017.json"
    vocab_path = "./checkpoints/vocab.pth"

    if os.path.isfile(vocab_path):
        print("[INFO] LOADING VOCABULARY ...")
        vocab = torch.load(vocab_path)
        coco_ds = CocoDS(images_path_train, annotations_path_train, vocab)
    else:
        print("[INFO] CREATING VOCABULARY ...")
        coco_ds = CocoDS(images_path_train, annotations_path_train)
        vocab = coco_ds.vocab
        torch.save(vocab, vocab_path)

    coco_val_ds = CocoTestDS(images_path_val, annotations_path_val, vocab)

    vocab_size = len(vocab)
    
    train_dl = DataLoader(coco_ds, batch_size=batch_size, shuffle=True)
    
    encoder = Encoder().to(device)
    decoder = Decoder(hidden_size, vocab_size, emb_size, attention_size, img_emb_size, num_channels, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                lr=encoder_lr)
    decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                lr=decoder_lr)
    
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        print(f"LOADED CHECKPOINT FROM EPOCH {start_epoch - 1}")

    pad = coco_ds.vocab["<pad>"]
    print(f"STARTING TRAINING WITH EPOCHS: {epochs - start_epoch} AND BATCH SIZE: {batch_size}")
    for epoch in range(start_epoch, epochs):
        print(f"STARTING TRAIN FOR EPOCH {epoch} ...")
        train_start_time = time()
        train_losses = train(train_dl, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, pad)
        train_time = time() - train_start_time
        print(f"TRAINING DONE: {train_time} seconds used.\nSTARTING VALIDATION ...")
        
        val_start_time = time()
        score, val_losses = validate(coco_val_ds, encoder, decoder, criterion, pad)
        val_time = time() - val_start_time
        print(f"VALIDATION DONE: {val_time} seconds used.\nSAVING MODEL CHECKPOINTS ...")
        
        save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, best_score, epoch, score > best_score)
        print("MODEL CHECKPOINTS SAVED\nSAVING MODEL LOSSES ...")
        
        save_losses(epoch, train_losses, val_losses, score, train_time, val_time)
        print("MODEL LOSSES SAVED")

        if score > best_score:
            print(f"NEW BEST BLEU SCORE: {score}")
            best_score = score
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch >= 0 and  epoch % email_freq == 0:
            print("SENDING UPDATE EMAIL ...")
            send_email_train_progress(best_score, epoch, score, epochs_since_improvement)

        if epochs_since_improvement > 0 and  epochs_since_improvement % 8 == 0:
            print("ADJUSTING LEARNING RATES ...")
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
            print("LEARNING RATES ADJUSTED")

        if epochs_since_improvement == 16:
            print("ENDING TRAINING : NO PROGRESS IN BLEU SCORE FOR 16 EPOCHS")
            break
    
    send_email_train_complete(best_score, epoch)
        
if __name__ == "__main__":
    main()