import yagmail
import os
import torch
from statistics import mean 
from constants import *

yag = yagmail.SMTP(sender, password)

save_path_last = f"checkpoints/last.pth"
save_path_best = f"checkpoints/best.pth"
save_path_losses = f"checkpoints/losses.pth"

def send_email_train_progress(best_score, epoch, score, epochs_since_improvement):
    yag.send(to=receiver, 
             subject=f"[TREINO VISION] PROGRESSO DE TREINO DE ÉPOCA {epoch}", 
             contents=f"Relatório de progresso de treino\n\nÉpoca: {epoch}\nMelhor Score: {best_score}\nScore Atual: {score}\nEpoques desde melhoria: {epochs_since_improvement}"
            )

def send_email_train_complete(best_score, epoch):
    yag.send(to=receiver, 
             subject="[TREINO VISION] FIM DE TREINO DA REDE", 
             contents=f"REDE TREINADA COM SUCESSO.\n\nÉpoca: {epoch}\nMelhor Score: {best_score}"
            )
    
def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, best_score, epoch, is_best=False):
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "encoder_optimizer": encoder_optimizer.state_dict(),
        "decoder_optimizer": decoder_optimizer.state_dict(),
        "best_score": best_score,
        "epoch": epoch
    }, save_path_last)

    if not is_best:
        return
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "encoder_optimizer": encoder_optimizer.state_dict(),
        "decoder_optimizer": decoder_optimizer.state_dict(),
        "best_score": best_score,
        "epoch": epoch
    }, save_path_best)

def save_losses(epoch, train_losses, val_losses, bleu_score, train_time, val_time):    
    train_loss = mean(train_losses)
    val_loss = mean(val_losses)
    
    losses = [{
                "epoch": epoch, 
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "bleu_score": bleu_score,
                "train_time":train_time,
                "val_time":val_time
             }]
    
    if os.path.exists(save_path_losses):
        old_losses = torch.load(save_path_losses)
        losses = old_losses + losses    
    
    torch.save(losses, save_path_losses)