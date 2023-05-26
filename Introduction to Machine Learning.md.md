## Introduction to Machine Learning

### What is Machine Learning?

Machine learning (học máy) là một lĩnh vực trong trí tuệ nhân tạo (AI) tập trung vào việc phát triển các thuật toán và mô hình máy tính có khả năng tự học và cải thiện hiệu suất theo thời gian. Phương pháp này cho phép máy tính "học" từ dữ liệu mà không cần được lập trình một cách cụ thể.

### Machine Learning Types

Có hai loại chính của machine learning: học có giám sát (supervised learning) và học không có giám sát (unsupervised learning). Trong học có giám sát, mô hình được huấn luyện bằng cách cung cấp cho nó các cặp dữ liệu đầu vào và đầu ra mong muốn, để nó có thể học cách ánh xạ từ đầu vào tới đầu ra. Trong học không có giám sát, mô hình phải tìm ra cấu trúc và mô hình dữ liệu mà không có thông tin về đầu ra mong muốn.

### Machine Learning Algorithms

Các thuật toán machine learning phổ biến bao gồm:

- Học có giám sát: Các thuật toán như Hồi quy tuyến tính (Linear Regression), Hồi quy Logistic (Logistic Regression), Máy vector hỗ trợ (Support Vector Machines - SVM), Cây quyết định (Decision Trees), Rừng ngẫu nhiên (Random Forests), Mạng nơ-ron (Neural Networks), và Gradient Boosting.

- Học không giám sát: Các thuật toán như Phân cụm K-means (K-means Clustering), Phân cấp (Hierarchical Clustering), Mô hình hỗn hợp Gaussian (Gaussian Mixture Models - GMM), và Phân tích thành phần chính (Principal Component Analysis - PCA).

- Học sâu (Deep Learning): Học sâu là một lĩnh vực con của machine learning, tập trung vào việc xây dựng và huấn luyện các mạng nơ-ron sâu (deep neural networks) có khả năng học và trích xuất đặc trưng tự động từ dữ liệu. Các kiến trúc phổ biến trong học sâu bao gồm Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN) cho xử lý hình ảnh, Mạng nơ-ron tuần tự (Recurrent Neural Networks - RNN) cho xử lý dữ liệu chuỗi, và Mô hình Transformer cho xử lý ngôn ngữ tự nhiên.

- Học tăng cường (Reinforcement Learning): Học tăng cường là một phương pháp học trong đó một hệ thống tương tác với môi trường và tự động học từ kinh nghiệm thông qua việc nhận phần thưởng hoặc hình phạt. Thuật toán nổi tiếng trong học tăng cường là Q-learning và Deep Q-Networks (DQN).

### Applications of Machine Learning

Để áp dụng machine learning, quá trình bao gồm các bước sau:

- Chuẩn bị dữ liệu: Tiền xử lý dữ liệu, bao gồm việc loại bỏ dữ liệu nhiễu, xử lý dữ liệu thiếu, và chia dữ liệu thành tập huấn luyện và tập kiểm tra.

- Chọn và huấn luyện mô hình: Chọn một thuật toán phù hợp và huấn luyện mô hình trên tập dữ liệu huấn luyện. Quá trình này bao gồm tối ưu hóa các thông số và tham số của mô hình để đạt được hiệu suất tốt nhất.

- Đánh giá mô hình: Sử dụng tập dữ liệu kiểm tra để đánh giá hiệu suất của mô hình. Đánh giá có thể dựa trên các độ đo như độ chính xác (accuracy), độ mất mát (loss), độ F1, hay ma trận nhầm lẫn (confusion matrix) tùy thuộc vào bài toán cụ thể.

- Điều chỉnh và tinh chỉnh: Nếu mô hình không đạt được hiệu suất mong đợi, có thể cần điều chỉnh lại thiết lập hoặc cải tiến quá trình huấn luyện. Điều này có thể bao gồm thay đổi thuật toán, tăng số lượng dữ liệu huấn luyện, hoặc tối ưu hóa tham số.

- Triển khai mô hình: Sau khi mô hình đã đạt được hiệu suất tốt, nó có thể được triển khai để sử dụng trong ứng dụng thực tế. Điều này có thể bao gồm việc tích hợp mô hình vào hệ thống hiện có hoặc phát triển các ứng dụng và dịch vụ mới.

## 5. Neural networks

Mạng neural (Neural networks) là một thuật toán trong lĩnh vực machine learning (học máy) được lấy cảm hứng từ cách hoạt động của hệ thống thần kinh sinh học. Mạng neural được sử dụng để học và nhận biết các mô hình phức tạp từ dữ liệu đầu vào.

Một mạng neural bao gồm các "neuron" nhân tạo được tổ chức thành các lớp (layers). Các lớp này kết nối với nhau thông qua các trọng số, và thông tin được truyền từ lớp đầu vào (input layer) qua các lớp ẩn (hidden layers) đến lớp đầu ra (output layer). Mỗi neuron tính toán đầu ra dựa trên tổ hợp tuyến tính của đầu vào và hàm kích hoạt phi tuyến (non-linear activation function).

Quá trình huấn luyện mạng neural bao gồm việc điều chỉnh các trọng số của mạng để tối thiểu hóa sai số giữa kết quả dự đoán và kết quả thực tế. Các phương pháp tối ưu hóa như Gradient Descent và Backpropagation thường được sử dụng để cập nhật trọng số.

Mạng neural có thể có kiến trúc đơn giản với chỉ một lớp ẩn hoặc kiến trúc phức tạp với nhiều lớp ẩn. Một số kiến trúc mạng neural phổ biến bao gồm mạng neural truyền thẳng (feedforward neural network), mạng neural hồi quy (recurrent neural network), và mạng neural chập (convolutional neural network).

Mạng neural có khả năng học và nhận biết các mô hình phức tạp, như phân loại hình ảnh, nhận dạng giọng nói, dịch máy và nhiều bài toán khác. Đánh giá hiệu suất của mạng neural thường được thực hiện bằng cách sử dụng các độ đo như độ chính xác (accuracy), độ mất mát (loss) và ma trận lỗi (confusion matrix).

Mạng neural đã trở thành một công cụ quan trọng trong lĩnh vực machine learning, và sự phát triển của các kiến trúc và thuật toán mới đã đóng góp đáng kể vào sự tiến bộ của lĩnh vực này.

## Kiến thức tổng quát về ML + ứng dụng

Một số khái niệm quan trọng trong ML bao gồm:

- Học có giám sát (Supervised Learning): Thuật toán học từ dữ liệu đã được gán nhãn, mục tiêu là xây dựng một mô hình dự đoán chính xác nhãn cho các dữ liệu chưa biết trước.

- Học không giám sát (Unsupervised Learning): Thuật toán học từ dữ liệu không được gán nhãn, mục tiêu là tìm ra cấu trúc, mẫu, hoặc nhóm các dữ liệu dựa trên đặc trưng tự nhiên của chúng.

- Học bán giám sát (Semi-supervised Learning): Kết hợp cả hai loại học, sử dụng một phần dữ liệu có nhãn và một phần không có nhãn để xây dựng mô hình.

- Học tăng cường (Reinforcement Learning): Mô hình học từ môi trường thông qua việc thử và sai, dựa trên phản hồi và phần thưởng nhận được từ môi trường.

Ứng dụng của ML rất đa dạng và phong phú. Dưới đây là một số ví dụ:

- Nhận dạng hình ảnh: ML được sử dụng để nhận dạng và phân loại đối tượng, khuôn mặt, hoặc biểu đồ trong ảnh.

- Xử lý ngôn ngữ tự nhiên: ML được áp dụng để xử lý, phân tích và tổng hợp ngôn ngữ tự nhiên, bao gồm dịch máy, phân loại văn bản, và phân tích cảm xúc.

- Hệ thống gợi ý: ML được sử dụng để tạo ra hệ thống gợi ý sản phẩm, phim, âm nhạc, dựa trên lịch sử và sở thích cá nhân của người dùng.

- Dự báo và dự đoán: ML được sử dụng để dự đoán xu hướng

- Phân tích dữ liệu và dự báo kinh doanh: ML được sử dụng để phân tích dữ liệu kinh doanh, dự báo xu hướng thị trường, và tối ưu hóa quyết định kinh doanh.

- Xử lý giọng nói và nhận dạng giọng nói: ML được áp dụng trong các ứng dụng như nhận dạng giọng nói, chuyển đổi giữa giọng nói và văn bản, và xử lý giọng nói tự nhiên.

- Xử lý dữ liệu y tế: ML được sử dụng để phân tích hồ sơ bệnh nhân, dự đoán bệnh tật, hỗ trợ chẩn đoán y tế, và tối ưu hóa quy trình chăm sóc sức khỏe.

- Xử lý dữ liệu tài chính: ML được sử dụng để phân tích dữ liệu tài chính, dự đoán rủi ro tài chính, phát hiện gian lận tài chính, và quản lý danh mục đầu tư.

- Xử lý dữ liệu vật liệu và khoa học: ML được sử dụng để dự đoán tính chất vật liệu, tối ưu hóa công thức hóa học, và tìm kiếm và phân loại dữ liệu khoa học.

- Xử lý dữ liệu vận chuyển và logistics: ML được sử dụng để tối ưu hóa lộ trình giao hàng, dự đoán thời gian giao hàng, và quản lý dữ liệu vận chuyển và kho hàng.
