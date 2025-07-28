from flask import Flask, render_template, request
import re
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
# from gensim.models import KeyedVectors
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import warnings
import os
import pytesseract
from PIL import Image
warnings.filterwarnings('ignore')
from Model.processing_data import processing_data
from record_En.recording import spechToText
#from OCR_img_to_Text.image_to_text import Img_to_text


app = Flask(__name__)

df = processing_data()
# Định nghĩa các lớp của mô hình
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình với cùng cấu hình như khi huấn luyện
SRC_VOCAB_SIZE = 21199
TGT_VOCAB_SIZE = 14929
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROP_OUT = 0.1

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROP_OUT)

# Tải trọng số vào mô hình
transformer.load_state_dict(torch.load("Model/save_model/viEn_transformer.pth", map_location=DEVICE))
transformer = transformer.to(DEVICE)

# Chuyển mô hình về chế độ đánh giá
transformer.eval()

# Create source and target language tokenizer.
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'

# Place-holders
token_transform = {}
vocab_transform = {}

# Tokenize for vietnames by underthesea
def vi_tokenizer(sentence):
    tokens = word_tokenize(sentence)
    return tokens

token_transform[SRC_LANGUAGE] = get_tokenizer('basic_english')
token_transform[TGT_LANGUAGE] = get_tokenizer(vi_tokenizer)

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    for index,data_sample in data_iter:
        yield token_transform[language](data_sample[language])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = df.iterrows()
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# def vi_tokenizer(sentence):
#     tokens = word_tokenize(sentence)
#     return tokens
#
# token_transform[SRC_LANGUAGE] = get_tokenizer('basic_english')
# token_transform[TGT_LANGUAGE] = get_tokenizer(vi_tokenizer)

# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                               vocab_transform[ln], # Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


# # Ví dụ sử dụng để dịch một câu
# def dich ( src_sentence = '0' ):
#     src_sentence = input('nhập : ')
#     translated_sentence = translate(transformer, src_sentence)
#     print(src_sentence, "   Translated Sentence:", translated_sentence)



# # Hàm dịch
# def translate(model: torch.nn.Module, src_sentence: str):
#     model.eval()
#     src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
#     num_tokens = src.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")




# # bỏ các ký tự đặc biệt 
# def clean_text(text: str) -> str:
#     # Define the special characters to remove
#     special_chars = r'[!@#$%^&*_\+]'
#     # Remove the defined special characters
#     cleaned_text = re.sub(special_chars, '', text)
#     return cleaned_text

UPLOAD_FOLDER = 'uploads'

# Kiểm tra và tạo thư mục nếu chưa tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ERT = False

# xử lý data 
def process_extracted_text(text):
    lines = text.split('\n')
    final_text = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  # Chỉ xử lý các dòng không rỗng
            # Kiểm tra xem dòng cuối cùng có kết thúc bằng dấu chấm không
            if final_text and final_text[-1].endswith('.'):
                final_text.append(stripped_line)  # Xuống dòng sau dấu chấm
            else:
                if final_text:
                    final_text[-1] += ' ' + stripped_line  # Nối với khoảng trắng
                else:
                    final_text.append(stripped_line)  # Dòng đầu tiên

    # Nối các dòng cuối cùng thành một chuỗi
    return '\n'.join(final_text)

@app.route('/', methods=['GET', 'POST'])
def index():
    global ERT
    src_sentence = ""
    translated_sentence = ""

    if request.method == 'POST':
        # Check if the microphone button was clicked
        if 'mic_button' in request.form:
            ERT = not ERT
            print(f"ERT state: {ERT}")

            if ERT:
                ERT = True # Khi bắt đầu đoạn ghi âm ERT chuyển về True ( Màu đỏ )
                src_sentence = spechToText()  # Bắt đầu ghi âm
                ERT = False # khi hết ghi âm sẻ trở về màu bình thường ẺT bằng False 
                #print("------------>" , ERT)
            else:
                src_sentence = spechToText(Bien=ERT)
                #print('==================' , ERT)# Tắt ghi âm

        elif 'img_button' in request.form:
            # Nhận và xử lý ảnh
            if 'image' not in request.files:
                return 'No image uploaded', 400

            image_file = request.files['image']
            if image_file.filename == '':
                return 'No image selected', 400

            # Lưu ảnh tạm thời
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Nhận tọa độ cắt từ form
            x = int(float(request.form['cropX']))
            y = int(float(request.form['cropY']))
            width = int(float(request.form['cropWidth']))
            height = int(float(request.form['cropHeight']))

            # Mở ảnh và cắt
            img = Image.open(image_path)
            cropped_img = img.crop((x, y, x + width, y + height))

            # Trích xuất văn bản từ ảnh đã cắt
            src_sentence = pytesseract.image_to_string(cropped_img)

            # Xóa file tạm thời
            os.remove(image_path)

            # Xử lý văn bản để xuống dòng theo yêu cầu
            src_sentence = process_extracted_text(src_sentence)

        else:
            src_sentence = request.form['sentence']

        # Nếu người dùng nhập vào không có dấu '.' ở cuối thì thêm vào
        number_len_text = len(src_sentence)
        src_sentence = src_sentence.replace("|", "I")
        #print('----------------------------------------------' ,number_len_text)
        if number_len_text != 0:
            if src_sentence[number_len_text - 1] not in ['.']:
                src_sentence = src_sentence + '.'


        # Tách tùng câu có kí tự đặc biệt và dịch từng câu
        save_text = ''
        save_text_dich = ''
        for i in src_sentence:
            print('1-----',save_text_dich)
            print('2-----',i)
            if i in ['.', '!', '?',',',':','[',']','(',')']:
                if len(src_sentence) != 0 and len(save_text) != 0:
                    print('3-------------------------',i)
                    save_text = translate(transformer, save_text)
                    print("4==============" , save_text)
                save_text_dich = save_text_dich + save_text + i    
                save_text = ''
            else:
                if i != "'" :
                    save_text += i  

        
        # Làm sạch câu đầu vào
        cleaned_sentence = src_sentence # clean_text(src_sentence)
        translated_sentence = save_text_dich
        # Chuyển những từ cái đầu câu thành in hoa
        translated_sentence= ". ".join([c.strip().capitalize() for c in translated_sentence.split(".")])

        # Dịch câu đã làm sạch
        # translated_sentence = translate(transformer, cleaned_sentence)

    return render_template('index.html', src_sentence=src_sentence, translated_sentence=translated_sentence)

if __name__ == '__main__':
    app.run(debug=True)
