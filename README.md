# Giải quyết vấn đề khởi động lạnh trong khuyến nghị khóa học trực tuyến bằng tăng cường dữ liệu dựa trên mô hình ngôn ngữ lớn

## 1. Giới thiệu

Dự án nghiên cứu ứng dụng mô hình ngôn ngữ lớn (LLM) để tăng cường thông tin khóa học, hướng đến giải quyết vấn đề khởi động lạnh (*cold-start*) trong hệ thống khuyến nghị khóa học trực tuyến (MOOC).

Các thành phần chính:
- **Dữ liệu:** MOOCCubeX.
- **LLM:** Qwen2.5-7B-Instruct, dùng để sinh hồ sơ sư phạm 5 chiều cho khóa học.
- **Mã hóa văn bản:** Sentence-BERT (`all-MiniLM-L6-v2`), tạo vector nhúng 384 chiều.
- **Đánh giá:** Ba kịch bản Cold-Item, Warm và Few-Shot User; so sánh với các phương pháp cơ sở Random và MostPop.

## 2. Cấu trúc thư mục

```text
├── data/
│   ├── course_llm/          # Dữ liệu khóa học, hồ sơ LLM và embeddings
│   ├── full_interactions/   # Dữ liệu tương tác học viên – khóa học
│   ├── raw_data_processed/  # Dữ liệu đã tiền xử lý (.csv)
│   └── splits/              # Các tập train/validation/test
├── src/
│   ├── data/                # Notebook kết hợp dữ liệu và chia tập
│   ├── eval/                # Notebook đánh giá và kiểm thử
│   ├── prompts/             # Prompt sinh hồ sơ sư phạm
│   └── course_profile.py    # Sinh hồ sơ LLM và mã hóa văn bản
├── README.md
└── requirements.txt
```

## 3. Cài đặt

Yêu cầu Python 3.10 trở lên. GPU hỗ trợ CUDA được khuyến nghị khi chạy Qwen2.5-7B-Instruct.

Cài đặt các thư viện cần thiết:

```bash
pip install -U bitsandbytes accelerate transformers sentence-transformers scipy pandas pyarrow numpy torch
```

> TODO: Cập nhật `requirements.txt` với các phiên bản thư viện đã kiểm thử để bảo đảm khả năng tái lập môi trường.

## 4. Hướng dẫn thực thi

Thực hiện các bước theo thứ tự sau từ thư mục gốc của dự án.

### Bước 1. Tiền xử lý dữ liệu

Chạy notebook `src/data/01-combine-data.ipynb` để kết hợp dữ liệu, sau đó chạy `src/data/02-split-data.ipynb` để tạo các tập dữ liệu cho ba kịch bản đánh giá.

Thiết lập lấy mẫu 100.000 học viên với seed `2026`, không lọc k-core.

### Bước 2. Sinh hồ sơ và mã hóa khóa học

Tệp thực thi chính: `src/course_profile.py`.

**a. Mã hóa văn bản khóa học gốc**

```bash
python src/course_profile.py \
    --data_path "data/course_llm/course_data.parquet" \
    --encode_raw_only \
    --encoder_name "sentence-transformers/all-MiniLM-L6-v2" \
    --output_dir "data/course_llm"
```

**b. Sinh hồ sơ LLM**

Chia 3.769 khóa học thành hai shard để xử lý riêng. Ví dụ chạy shard đầu tiên:

```bash
python src/course_profile.py \
    --data_path "data/course_llm/course_data.parquet" \
    --prompt_path "src/prompts/prompt_template.txt" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --num_courses 3769 \
    --shard 0 --num_shards 2 \
    --max_new_tokens 300 --single_gpu \
    --output_dir "data/course_llm"
```

Chạy shard thứ hai bằng cách đổi `--shard 0` thành `--shard 1`.

**c. Gộp hồ sơ và tạo embeddings**

```bash
python src/course_profile.py \
    --data_path "data/course_llm/course_data.parquet" \
    --merge_shards --num_shards 2 \
    --encoder_name "sentence-transformers/all-MiniLM-L6-v2" \
    --output_dir "data/course_llm"
```

**d. Đánh giá khả năng suy luận tiên quyết bị ẩn**

```bash
python src/course_profile.py \
    --data_path "data/course_llm/course_data.parquet" \
    --prompt_path "src/prompts/prompt_template.txt" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --masked_prereq_eval \
    --eval_sample_size 200 --eval_seed 2026 \
    --max_new_tokens 300 --single_gpu \
    --output_dir "data/course_llm"
```

### Bước 3. Đánh giá hệ thống

Chạy notebook `src/eval/03-evaluate.ipynb` để thực hiện kiểm thử và đánh giá các phương pháp cơ sở Random, MostPop trên ba kịch bản.

## 5. Kết quả đầu ra

Các tệp chính được lưu trong `data/course_llm/`:

| Tệp | Mô tả |
|---|---|
| `full_3769_profiles.json` | Hồ sơ sư phạm 5 chiều của 3.769 khóa học |
| `llm_profile_embeddings.npy` | Vector nhúng hồ sơ LLM, kích thước `(3769, 384)` |
| `raw_course_text_embeddings.npy` | Vector nhúng văn bản khóa học gốc, kích thước `(3769, 384)` |
| `embeddings_course_index.json` | Ánh xạ chỉ số khóa học và thông tin liên quan |
| `masked_prereq_200_eval_report.json` | Báo cáo đánh giá trên 200 khóa học có tiên quyết bị ẩn |

Các tệp báo cáo benchmark và phân tích bổ sung được tạo trong quá trình thực thi.

## 6. Trích dẫn dữ liệu

Dự án sử dụng bộ dữ liệu MOOCCubeX:

```bibtex
@inproceedings{yu2021mooccubex,
  title={MOOCCubeX: A large knowledge-centered repository for adaptive learning in MOOCs},
  author={Yu, Jifan and Wang, Yuquan and Zhong, Qingyang and Luo, Gan and Mao, Yiming and Sun, Kai and Feng, Wenzheng and Xu, Wei and Cao, Shulin and Zeng, Kaisheng and others},
  booktitle={Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
  pages={4643--4652},
  year={2021}
}
```
