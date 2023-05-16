# Machine Learning 2023

Tóm tắt kiến thức Machine Learning

## Introduction to Machine Learning

Machine learning (học máy) là một lĩnh vực trong trí tuệ nhân tạo (AI) tập trung vào việc phát triển các thuật toán và mô hình máy tính có khả năng tự học và cải thiện hiệu suất theo thời gian. Phương pháp này cho phép máy tính "học" từ dữ liệu mà không cần được lập trình một cách cụ thể.

Có hai loại chính của machine learning: học có giám sát (supervised learning) và học không có giám sát (unsupervised learning). Trong học có giám sát, mô hình được huấn luyện bằng cách cung cấp cho nó các cặp dữ liệu đầu vào và đầu ra mong muốn, để nó có thể học cách ánh xạ từ đầu vào tới đầu ra. Trong học không có giám sát, mô hình phải tìm ra cấu trúc và mô hình dữ liệu mà không có thông tin về đầu ra mong muốn.

Các thuật toán machine learning phổ biến bao gồm:

- Học có giám sát: Các thuật toán như Hồi quy tuyến tính (Linear Regression), Hồi quy Logistic (Logistic Regression), Máy vector hỗ trợ (Support Vector Machines - SVM), Cây quyết định (Decision Trees), Rừng ngẫu nhiên (Random Forests), Mạng nơ-ron (Neural Networks), và Gradient Boosting.

- Học không giám sát: Các thuật toán như Phân cụm K-means (K-means Clustering), Phân cấp (Hierarchical Clustering), Mô hình hỗn hợp Gaussian (Gaussian Mixture Models - GMM), và Phân tích thành phần chính (Principal Component Analysis - PCA).

- Học sâu (Deep Learning): Học sâu là một lĩnh vực con của machine learning, tập trung vào việc xây dựng và huấn luyện các mạng nơ-ron sâu (deep neural networks) có khả năng học và trích xuất đặc trưng tự động từ dữ liệu. Các kiến trúc phổ biến trong học sâu bao gồm Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN) cho xử lý hình ảnh, Mạng nơ-ron tuần tự (Recurrent Neural Networks - RNN) cho xử lý dữ liệu chuỗi, và Mô hình Transformer cho xử lý ngôn ngữ tự nhiên.
- Học tăng cường (Reinforcement Learning): Học tăng cường là một phương pháp học trong đó một hệ thống tương tác với môi trường và tự động học từ kinh nghiệm thông qua việc nhận phần thưởng hoặc hình phạt. Thuật toán nổi tiếng trong học tăng cường là Q-learning và Deep Q-Networks (DQN).

Để áp dụng machine learning, quá trình bao gồm các bước sau:

- Chuẩn bị dữ liệu: Tiền xử lý dữ liệu, bao gồm việc loại bỏ dữ liệu nhiễu, xử lý dữ liệu thiếu, và chia dữ liệu thành tập huấn luyện và tập kiểm tra.
- Chọn và huấn luyện mô hình: Chọn một thuật toán phù hợp và huấn luyện mô hình trên tập dữ liệu huấn luyện. Quá trình này bao gồm tối ưu hóa các thông số và tham số của mô hình để đạt được hiệu suất tốt nhất.
- Đánh giá mô hình: Sử dụng tập dữ liệu kiểm tra để đánh giá hiệu suất của mô hình. Đánh giá có thể dựa trên các độ đo như độ chính xác (accuracy), độ mất mát (loss), độ F1, hay ma trận nhầm lẫn (confusion matrix) tùy thuộc vào bài toán cụ thể.
- Điều chỉnh và tinh chỉnh: Nếu mô hình không đạt được hiệu suất mong đợi, có thể cần điều chỉnh lại thiết lập hoặc cải tiến quá trình huấn luyện. Điều này có thể bao gồm thay đổi thuật toán, tăng số lượng dữ liệu huấn luyện, hoặc tối ưu hóa tham số.
- Triển khai mô hình: Sau khi mô hình đã đạt được hiệu suất tốt, nó có thể được triển khai để sử dụng trong ứng dụng thực tế. Điều này có thể bao gồm việc tích hợp mô hình vào hệ thống hiện có hoặc phát triển các ứng dụng và dịch vụ mới.
