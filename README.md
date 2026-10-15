Tên đề tài: Giải quyết vấn đề khởi động lạnh trong khuyến nghị khóa học trực tuyến bằng tăng cường dữ liệu dựa trên mô hình ngôn ngữ lớn

---

## Dữ liệu & Trích dẫn MOOCCubeX
Dữ liệu sử dụng trong đề tài được xây dựng dựa trên bộ dữ liệu công khai **MOOCCubeX**.
* **Trích dẫn mã nguồn gốc của bài báo công khai bộ dữ liệu MOOCCubeX:**
  ```bibtex
  @inproceedings{yu2021mooccubex,
    title={MOOCCubeX: A large knowledge-centered repository for adaptive learning in MOOCs},
    author={Yu, Jifan and Wang, Yuquan and Zhong, Qingyang and Luo, Gan and Mao, Yiming and Sun, Kai and Feng, Wenzheng and Xu, Wei and Cao, Shulin and Zeng, Kaisheng and others},
    booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
    pages={4643--4652},
    year={2021}
  }

## Cấu trúc thư mục
```text  .
  ├── data/
  │   |── course_llm/           # Hồ sơ LLM sinh ra
  |   |── full_interactions/    # Dữ liệu tương tác giữa học viên và khóa học
  │   |── raw_data_processed/   # Dữ liệu gốc đã tiền xử lý
  |   └── splits/               # Các tập train/val/test theo giao thức (seed: 2026)
  ├── src/
  │   ├── data/                 # Mã nguồn kết hợp dữ liệu và chia split theo 3 kịch bản
  │   ├── eval/                 # Thư mục chứa evaluator.py và các bài test đơn vị
  │   └── prompts               # Prompt sinh mô tả bằng LLM
  ├── README.md
  └── requirements.txt
