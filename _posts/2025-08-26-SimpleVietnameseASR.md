---
title: Simple Vietnamese ASR Model
layout: post
date: 2025-08-26
categories: [Blog, Tech]
tags: [speech-asr]
author: ltnghia
description: This post is about ASR model for Vietnamese language, using pre-trained Wav2Vec2 model from facebook and CTC loss function to optimize.
math: true
---

## Giới thiệu về bài toán
Vietnamese ASR là bài toán Speech-To-Text nhưng với ngôn ngữ đối tượng chính là Tiếng Việt. Bài toán nhận đầu vào là audio Tiếng Việt và kết quả là một bản phụ đề cho audio đó.

Những đặc điểm của Tiếng Việt:
- Tiếng Việt thuộc ngữ hệ Nam Á (Austroasiatic), có sự ảnh huởng từ các ngôn ngữ khác như Trung Quốc hay Pháp.
- Ngôn ngữ có sự phức tạp về thanh điệu (tonal) do đến 6 thanh điệu đối miền Bắc Việt Nam (ngang, sắc, huyền, hỏi, ngã, nặng), đối với Miền Trung và Miền Nam Việt Nam thì không có thanh ngang và không có các cụm phụ âm phức tạp như Tiếng Anh. 
- Cô lập (isolating)  tức là không biến đổi hình thái như số nhiều hay ở các thì ngữ pháp Tiếng Anh, ví dụ từ "ăn" có thể là "eat/eating/ate". 
- Đơn âm tiết, không ghép phức tạp, phonemic dự trên chữ Latin vớ diacritíc cho tones (ví dụ: â, ê, dâí thanh). Mối quan hệ chữ-âm nhất quán, dễ dự đoán phát âm của từ.
- Nhìn chung Tiếng Việt và Tiếng Anh đều giống nhau ở phần lớn các chữ cái, và các từ đều được phân tách rõ ràng bởi ký tư khoảng trắng (space).

Nên các mô hình ASR đã được huấn luyện trước với tập dữ liệu Tiếng Anh có thể sẽ không đáp ứng được cho các dữ liệu Tiếng Việt, bên cạnh đó dữ liệu Tiếng Việt cũng tương đối hạn chế.

## Dataset FOSD
FPT Open Speech Dataset là một dataset gồm khoảng 30 giờ âm thanh ghi âm giọn nói Tiếng Việt, được thu thập bởi FPT Corporation vào năm 2018.

Dataset chứa 25.921 bản ghi âm định dạng `.mp3`, đi kèm bảng phụ đề (`.txt`, mã hóa UTF-8) và thông tin về thời gian bắt đầu - kết thúc cho mỗi đoạn ghi âm. Có phiên bản Hugging Face (mirror) cho phép truy cập dễ dàng bằng Python thông qua `datasets`.

Bài toán sẽ sử dụng FOSD cho mô hinh nhận diện giọng nói Tiếng Việt. 

## Tiền xử lý audio, transcript của raw dataset và tạo vocab cho mô hình
Vì trong raw dataset thì các mẫu audio là các file `.mp3` và transcript cũng có những ký tự có thể gọi là "không thể phát âm" do nó không có một cách phát âm riêng. Ví dụ: ký tự '+' được đọc là 'dấu cộng', được cấu thành từ các ký tư trong bản chữ cái. 

Các bước chuẩn hóa audio và transcript và tạo vocab:
- Chuyển từng file `.mp3` về đúng định dạng đầu vao của mô hình Wav2Vec 2.0 là có phần mở rộng là `.wav` với sampling rate 16 kHz, sử dụng công cụ xử lý là `ffmpeg`.

```python
subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', mp3_file,
                '-acodec', 'pcm_s16le',
                '-ac', str(1),
                '-ar', str(16000),
                wav_file
            ])
```
- Ứng với từng file âm thanh được xử lý sẽ là một transcript, ta tiến hành sử dụng thư viện `re` là regular expression để loại bỏ nó ra khỏi text, kể cả punctuation và các ký tự đặc biệt khác như `\r`, `\n`, tiến hành thay thế bằng ký tự rỗng `''`. 
- Trong transcript có những ký tự `-N` tượng trưng cho đoạn âm thanh đó không có transcript, sẽ đuợc thay thế bằng ký tự khác (giả sử `=`), và ký tự `=` sau đó sẽ được thay thế bằng token `[UNK]` trong vocab. Ký tự khoảng trắng cũng sẽ được thay thế bằng ký tư `|`, vì Wav2Vec2 được pre-trained vói ký tự `|` như là ký tự phân tách giữa các từ với nhau.
```python
script = re.sub(r"(-N)", "UNK", script)
pattern = f"(\\r\\n|\\r|\\n|\\t|–|”|“|[{re.escape(string.punctuation)}]|[0-9])"
script = re.sub(pattern, '', script)
script = re.sub(r"(UNK)", "=", script)
```
- Tất cả được chuyển về dạng lowercase và tách thành từng ký tự riêng rẽ được đưa vào vocab ứng với từng id riêng của nó. 

```python
list_transcript = concat_transcript(manifest_path)
all_text = " ".join(list_transcript).lower()
vocab = sorted(set(all_text))
vocab_dict = {c: i for i, c in enumerate(vocab)}
vocab_dict['|'] = vocab_dict[' '] # Wav2Vec2CTCTokenizer from Hugging Face default using | as word_delimiter_token. Not overlap with padding with ' '
    
if vocab_dict.get('='):
vocab_dict['[UNK]'] = vocab_dict['=']
del vocab_dict['=']

vocab_dict['[UNK]'] = 95
del vocab_dict[' ']

with open(vocab_path, 'w', encoding='utf-8') as f:
	json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
```


Kết quả trong `vocab.json`:
```json
{
  "a": 1,
  "b": 2,
  "c": 3,
  ...
  "à": 27,
  "á": 28,
  "â": 29,
  "ã": 30,
  "è": 31,
  "é": 32,
  ...
  "|": 0,
  "[UNK]": 95
}
```

## Wav2Vec2FeatureExtractor
Nhiệm vụ chính của thành phần này là chuẩn bị dầu vào dạng (raw audio) cho model chính. Feature Extractor tiến hành chuẩn hóa lại sampling rate về chuẩn của Wav2Vec 2.0, padding các audio trong một batch về cùng một dộ dài để đưa về tensor audio của batch, cũng có thể consider đến việc chuẩn hóa (như zero-mean, unit-variance)
- **Input:** Một batch của các mảng numpy/tensor 1D (raw waveform)
- **Output:** Tensor đặc trưng đã được chuẩn hóa, sẵn sàng đưa vào Encoder cả Wav2Vec2.

## Wav2Vec2CTCTokenizer
Nhiệm vụ xử lý chuỗi ký tự transcript thành chuỗi các token ID thông qua vocab đã được xây dựng từ trước. Wav2Vec2 cho ASR, tokenizer thường là CTC tokenizer (character-level): mapping từng ký tự trong text sang ID là số, thêm các ký tự đặc biệt cho padding hay là blank token.

- **Input:** text transcript (ví dụ "xin chào")
- **Output:** mảng ID của text transcript (ví dụ "24 9 14 0 3 8 27 15" theo như vocab đã được xây dựng), ngược lại cũng có thể decode từ ID sequence sang text.

## Wav2Vec2Processor
Nhiệm vụ gói chung Feature Extractor và Tokenizer vào môt đối tượng duy nhất để dễ xử lý hơn. Không cần phải phải riêng lẻ extractor và tokenizer, chỉ dùng `processor`, nhưng nếu cần thì vẫn có thể sử dụng riêng lẻ.
```python
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True, do_normalize=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

encoded_input = self.processor(audio=audio.squeeze(), sampling_rate=16000, text=text)
encoded_input["input_length"] = len(encoded_input["input_values"][0])
encoded_input["label_length"] = len(encoded_input["labels"])
```
## Wav2Vec2Model
Wav2Vec 2.0 được cải tiến từ mô hình Wav2Vec của facebook, mục tiêu của Wav2Vec 2.0 là học các biểu diễn âm thanh chất lượng trực tiếp từ raw audio, không phụ thuộc vào nhãn của dữ liệu (self-supervised learning). Thường sử dụng cho các downstream tasks như ASR hay Setiment Classification hay Speech Classification. Cho phép fine-tuning hiệu quả với dataset có nhãn nhỏ.

Mô hình Wav2Vec 2.0 được huấn luyện thông qua việc mô hình sẽ mask một phần input ở latent space (sau bước feature encoder), sau đó giải quyết nhiệm vụ contrastive loss trên các quantization của latent representation. Loss function bao gồm:
- Contrastive loss: Tính toán sự tương đồng giữa các biểu diễn bị mask và các quantization thực. đồng thời giảm tương đồng với các distractors.
- Diversity loss: Khuyến khích các quantization đa dạng hơn (như Regularization).
## Quy trình huấn luyện
1. Raw audio được xử lý qua CNN để tạo ra các latent feature vectors.
2. Mask một phần latent feature vectors để đưa vào Transformer dự đoán các phần bị mask.
3. Trước khi mask thì các vector này đã được đưa qua khối quantization để lượng tử hoá các vector thành các mã rời rạc sử dụng Gumbel-Softmax và codebooks. Ben cạnh đó còn có các distractor (các mẫu từ cùng một batch hoặc ngẫu nhiên).
4. So sánh các phần bị mask được Transformer dự đoán với các quantization thực tế để tính toán mất mát và lặp lại quá trình tối ưu trên toàn bộ dataset. Giúp mô hình học đwuọc khác biệt giữa âm thanh gốc và biến đổi. [Chi tiết về mô hình Wav2Vec 2.0](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)

Nhưng khi sử dụng mô hình pre-trained Wav2Vec 2.0 thì ta chỉ cần lấy kết quả cuối cùng từ context network (tức là Transformer của model) và xem như đó là các feature vectors của input audio.

## Chuẩn bị dataset và dataloader cho mô hình
Khi dùng `Dataset` và `DataLoader` của pytorch thì phải tự thiết kế lại để phù hợp với input batch của mô hình. Với từng example đơn lẻ trong dataset thì sẽ bao gồm một raw audio và một text đi kèm. Raw audio có thể được chuẩn hóa về sampling rate hoặc zero-mean và unit-variance, text thì cần phải được encode sang chuỗi các ID trong vocab.

Tuy nhiên `DataLoader` của pytorch lại không hỗ trợ xử lý và gom batch các example theo yêu cầu củ Wav2Vec2 nen ta cũng cần phải can thiệp để định nghĩa ra một `DataCollator` để đáp ứng.

```python
@dataclass
class DataCollatorCTCWithPadding: # use to pass into colatte_fn of DataLoader to join single samples into batch
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: list[dict[
                        Union[list[int], torch.Tensor],
                        str
                    ]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and 
        # need different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        labels_batch = [feature["labels"] for feature in features]

        input_lengths = torch.tensor(data=[feature["input_length"] for feature in features], dtype=torch.long)
        label_lengths = torch.tensor(data=[feature["label_length"] for feature in features], dtype=torch.long)

        batch = self.processor.pad(input_features=input_features, padding=self.padding, return_tensors='pt')

        batch["labels"] = labels_batch
        batch["input_lengths"] = input_lengths
        batch["label_lengths"] = label_lengths

        return batch
```

Khi đó một batch của dataset sẽ là một dictionary của `input_values`, `attention_mask`, `labels`, `input_lengths`, `label_lengths`. Trong đó:
- `input_values`: $(N, T)$. Tensor của các raw audio đã được padding theo mode `longest`.
- `attention_mask`: $(N, T)$. Tensor của chuỗi `0` và `1`, với `1` là giá trị thực với vị trí tương ứng trong tensor `input_values`, `0` là vị trí được padding. Ý nghĩa của tensor này đánh dấu cho Wav2Vec2 nhận biết để không tính toán với các giá trị được padding.
- `labels`: List của các chuỗi đã được tokenize, không cần phải cùng độ dài do dùng CTC thủ công.
- `input_lengths`: $(N,)$. Tensor 1D của độ dài thực tế các audio trong `input_values`.
- `label_lengths`: $(N,)$. Tensor 1D của độ dài thực tế các ID sequence trong `labels`.

Kết quả trả về của `Wav2Vec2Model` là môt tensor có chiều $(N, T, H)$, với $H$ là số hidden unit của layer cuối cùng của mô hình. Cần phải chuyển vị lại thành $(T,N,H)$ để phù hợp với yêu cầu của CTC loss.

## Objective Function
Connectional Temporal Classification được tìm ra thông qua các vấn đề như việc không biết chính xác ký tự nào tương ứng với đoạn nào trong raw audio và không thể gán nhãn một cách thủ công cho từng đoạn đối với dataset rất lớn.

CTC là thụâ toán gán giá trị xác suất cho output $Y$ khi cho biết bất cứ input $X$ nào. Ưu điểm của CTC là size của $X$ và $Y$ không cần phải match nhau. Ví dụ với Speech Recognition, input là raw audio chứa 16k samples cho mỗi giây (sr=16 kHz), nhưng trong một giây ta chỉ có thể nói được vài ký tự. Vì vậy ASR đang cố gắng mapping lượng lớn input vào một output có size nhỏ hơn rất nhiều.

![alt text](/assets/images/26-08-2025_post/ctc.png)

Để hiểu rõ hơn về CTC thì ta xét với 3 khía cạnh:

### Collapsing
- Vì output có kích thước nhỏ hơn nhiều so với input, ta tiến hành phân loại cơ bản trên input. Chia input thành từng frame bằng nhau, có thể là 400 samples với 16kHz sampling rate thì tầm 25 ms.  Khi đó cần classify từng frame thành các characters có sẵn trong vocab. Ví dụ đối với tiếng Anh thì ta có tất cả alphanumeric characters và ột số tokens đặc biệt (không phải ký tự đặc biệt).

- Trong ASR vocab, ta không đưa các ký tự đặc biệt vào vì nó là no sound, ví dụ `.`, `!`, `?`. Nhưng ký tự `'` có thể xác định như trong `It's`, nên nó có thể xuất hiện.

- Output sẽ cho một chuỗi các ký tự, khi đó kết hợp về mặt logic của CTC sẽ tham gia. Ý tưởng là nó sẽ kết hợp liên tục, và có thể lặp lại các ký tự. Ví dụ `hhhhiiii` và có thể viết thành `hi`.

- Ta phải xử lý 2 trường hợp đặc biệt: (1) Có thể có ký tự space giữa hai words; (2) Có thể có những ký tự được phép lặp lại nhiều lần trong một một từ đúng chính tả, như ký tự `l` trong `hello`. Để xử lý ta có thể thêm hai token đặc biệt vào vocab: `[BRK]` và `[SEP]` lần lượt cho từng case.

- Ví dụ như đối với `hello` có thể decoded nếu ta có classification output như là `hhelll[SEP]llo`. Từ đó ta có thể thực hiện nhiệm vụ phân loại đơn giản ở output layer và sau đó sử dụng CTC để decoding logic phần còn lại. Nhưng how to teach model to predict these output?

Thuật toán nhìn chung sẽ theo flow như sau: (1) Kết hợp liên tục các ký tự; (2) Remove tokens `[SEP]`; (3) Thay thế `[BRK]` tokens với ký tự space.

### Relevant Paths
- Cho mỗi time step, output sẽ là một phân phối xác suất cho mỗi ký tự trong vocab (giả sử sử dụng softmax).
- Giả sử ta có 3 time steps và 3 ký tự khác nhau trong `vocab=[h, i, [SEP]]`, khi đó ta có thể có $3^3 = 27$ paths để chọn.

![alt text](/assets/images/26-08-2025_post/possible_path.png)
- Tính chất thú vị của CTC là có thể có nhiều paths khả dĩ. CTC transcibe `hi`từ các chuỗi sau: `hhi`, `hii`, `h[SEP]i`, `hi[SEP]` hay là `[SEP]hi`. Khi đó tỷ lệ paths đúng đã là $5/27$.
- Ta muốn phạt các path incorrect và tăng tỷ lệ cho path đúng, mong muốn này có thể thực hiện bằng hai cách:
    - Huấn luyện để tăng tỷ lệ cho ký tự trong path đúng ở timestep cụ thể. Trong ví dụ trên, ở timestep đầu tiên thì tăng tỷ lệ cho `h` và `[SEP]`, lặp lại thao tác này cho các timestep khác. Nhưng cách tiếp cận này bị nhược điểm lớn là nó train theo mức độ của timestep, không phải path level (tức là ứng với từng timestep, ta huấn luyện nó học được tỷ lệ các ký tự ứng với timestep đó). Do đó tỷ lệ ở mỗi bước có thể được cải thiện, nhưng về tổng thể thì output paths có thể không phải là relevant.
    - Consider đến context of the path by using models like RNN (nếu được thì Transformer), có thể tính xác suất theo từng bước liên quan đến xác suất tổng thể của toàn bộ path. Nhân xác suất ở tất cả các bước trong relevant path $ \prod_{t=1}^T p_t(a_t \mid X)$ và tổng xác suất của path của tất cả các relevant paths ( $ \sum_{A\in \mathcal{A}_{X,Y}} $ ). Khi đó nó sẽ cho chúng ta xác suất có điều kiện CTC, khi đó ta cần tối đa hóa:
    
    $$ P(Y|X) = \sum_{A\in \mathcal{A}_{X,Y}}\prod_{t=1}^Tp_t(a_t|X) $$
    
Trong đó: 
- $ \mathcal{A}_{X,Y} $ là tập hợp tất cả path $A$ mà khi decoding hoàn chỉnh ra được $Y$; 
- $ p_t(a_t \mid X)$ xác suất mô hình dự đoán ký hiệu $ a_t $ tại timestep $t$; 
- $ \prod_{t=1}^T p_t(a_t \mid X) $ xác suất của cả path $ A $ (giả định các bước độ lập có điều kiện trên $X$ - đúng thật, vì ta chỉ dựa vào $ X $ để tính toán), sau đó tổng các xác suất các path có thể tạo ra $Y$.

Tại sao cần RNN hoặc mô hình có ngữ cảnh: thay vì dự đoán ký tự từng bước độ lập hoàn toàn thì ta dùng RNN hoặc Transformer để lấy ngữ cảnh toàn chuỗi, rồi mới ước lượng $p_t(a_t\mid X)$ cho từng timestep. Đây là optinal cho tính $p_t(a_t\mid X)$, tức là bước xử lý trước đó để tính xác suất.

Tính chất này của CTC giúp model học được cách biểu diễn mặc dù không cần perfect data annotations - cái mà ta cần phải gán nhãn cho từng tokens riêng biệt của input. Ta chỉ cần input audio stream và expected output transcription và CTC loss sẽ lo hết phần còn lại. Framework như Pytorch cũng đã hỗ trợ CTC loss.

### Inference
Sau khi huấn luyện model, chúng ta muốn nó hoạt động trong quá trình inference. Nhưng do ta tối ưu rất khác với cách chọn transcript đúng của paths, nên ta buộc phải chọn một.
- Nếu chọn token theo kiểu time step level thì nó sẽ dẫn đến tình trạng xác suất cao nhưng không đúng. Do quá trình train ta đã áp dụng theo kiểu sum các relevant path lại (dựa vào ngữ cảnh, nên gen theo timestep là chưa chính xác với model), nếu sum các relevant path lại thì sẽ có xác suất sao hơn. Nên nếu dự đoán theo kiểu timestep level sẽ dẫn đến các issues khác.
- Beam search.
- Utilize LM like n-gram LM or Neural LM (more powerful but more complicated to implement and slower). Nhưng mức độ cải thiện không đáng kể so với n-gram.


## Kết quả WER thu được trên tập test.
Sau quá trình huấn luyện cho đến khi loss "có vẻ không còn giảm nữa" thì tạm dừng và sử dụng các tham số đã được huấn luyện để kiểm tra trên tập test. Vì Tiếng Việt các từ được phân cách rõ ràng bởi khoảng trắng, nên sẽ ưu tiên sử dụng WER (word error rate) để tính toán mức độ hiệu quả của mô hình. Có thể so sánh giữa các epoch đầu hoặc so với WER của các SOTA.

Tập test được phân chia với tỷ lệ 20% trên toàn bộ dataset. Tất cả đều được xáo trộn ngẫu nhiên trước khi chia nên sẽ không gặp phải vấn đề tập test hay train sẽ bias về một giọng đọc duy nhất.

| Epochs       | WER (%)       |
|:-----------|:----------:|
| 05    |     869.27   |
| 20    |     891.21   |
| 55    |     8.9992   |

Nhận xét về chuỗi dự đoán từ mô hình:
- WER cho thấy có vẻ mô hình đang bị overfit trong khi loss giảm nhưng kết quả eval lại tăng... CTC thường tối ưu hóa ở mức phoneme hoặc character, trong khi WER đo lường lỗi ở mức từ, do đó có thể giảm loss sẽ không đảm bảo giảm WER, vì mô hình có thể cải thiẹn dự đoan ở mức sub-word nhưng không chuyển hóa tốt thành từ hoàn chỉnh.
- Cũng có thể xuất phát từ việc model quá tập trung vào tối ưu hóa các phoneme hoặc character riêng lẻ mà không chú ý đến ngữ cảnh từ vựng hoặc ngôn ngữ.
- Số lượng từ đúng hoàn toàn trong câu rất ít. Trong câu số từ đúng toàn vẹn cũng xuất hiện với tần suất rất thấp. Giả sử sử dụng tham số với epoch 55.
```text
người mang con nhỏ người thì vác tải gạo người nào người ấy lỉnh kỉnh đồ đạc|ười mà n cô n nh người thi vá tả g ười n ườ n n cứế độ đ
```
- Đa số các từ đều thiếu phụ âm cuối. Ví dụ `một` sẽ được dư đoán đa số là `mộ`. Điều này có thể được giải thích qua việc Tiếng Việt không phát âm các phụ âm cuối như Tiếng Anh, do đó để đúng từ này thì ta chỉ có thể dựa vào từ điển hoặc lỗi chính tả để sửa. CÒn trong âm thanh thì ta không có cơ sở để dự đoán nó.
- Có thể mô hình LM chưa được tích hợp tốt hoặc chưa được huấn luyện đủ trong giai đoạn decoding (beam search), đặc biệt khi mô hình acoustic cải thiện (loss giảm) nhung LM không theo kịp.
- Gần như tất cả các câu đúng đa số là về các phụ âm đầu của các từ trong câu, cac vần phía sau thì không thường thấy xuất hiện kèm theo.
- Nhưng kết quả tích cực ở chỗ các dấu phân tách từ được dự đoán khá đúng trong câu theo raw audio.
- Nhìn chung kết quả có được rất tệ.

### Nguyên nhân tiềm năng dẫn đến kết quả tệ của mô hình
- Về dataset: FOSD là một dataset được xem là nhỏ trong các tâp dataset ở lĩnh vực Speech, bên cạnh đó giọng của người nói trong dataset không được đa dạng và chất lượng giọng nói không thật sự cao nên cũng ảnh đến cách mà mô hinh trích xuất đặc trưng từ đó. Đặc biệt dữ liệu không đa dạng.
- Về vocab: Vocab Tiếng Việt được tạo ra như cách đề xuất ở phần trước cũng chưa thật sự là tối ưu. Theo bảng chữ cái Tiếng Việt thì chỉ bao gồm các nguyên âm như `a, ă, â, e, ê...`, còn dấu thanh thì được tách riêng và sẽ được kết hợp với nguyên âm sau đó. Có thể điều này cũng gây khó khăn cho mô hình trong việc nhận diện các nguyên âm khó như 'ắ, ậ, ỗ...'.
- Về level tokenizer: theo nhận xét của những dự án ASR về tiếng Việt trên Internet thì mức độ hiệu qủa của character-level tokenization thông thường sẽ là thấp nhất so với phoneme-level hay là sub-word và word level tokenization.
- Có thể xuất phát từ Learning rate hoặc scheduler không phù hợp dẫn đến việc mô hình khó học được các đặc trưng phức tạp ở giai đoạn sau.

Repo github của mini project này sẽ được cập nhật sau...




















































