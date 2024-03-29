# Regression

## What is regression?

Hồi quy (Regression) là một phương pháp trong machine learning (học máy) được sử dụng để dự đoán giá trị của một biến mục tiêu dựa trên các biến đầu vào. Nó giúp tìm hiểu mối quan hệ và ánh xạ giữa các biến để xây dựng một mô hình dự đoán.

Trong hồi quy, chúng ta tìm kiếm một hàm số hoặc mô hình có thể mô tả được mối quan hệ giữa biến đầu vào (x) và biến mục tiêu (y). Mô hình hồi quy cố gắng tìm ra mối quan hệ này bằng cách điều chỉnh các tham số của mô hình để tạo ra dự đoán chính xác nhất có thể.

Có nhiều loại mô hình hồi quy khác nhau, nhưng một trong những mô hình phổ biến là hồi quy tuyến tính (Linear Regression). Trong hồi quy tuyến tính, mô hình dự đoán một giá trị đầu ra liên tục dựa trên một tập hợp các biến đầu vào và trọng số tương ứng với mỗi biến. Mục tiêu là tìm ra bộ trọng số tối ưu để mô hình có thể dự đoán chính xác giá trị đầu ra.

Quá trình huấn luyện một mô hình hồi quy thường bao gồm việc tìm kiếm bộ trọng số tối ưu bằng cách sử dụng phương pháp tối ưu hóa. Phương pháp phổ biến nhất là phương pháp bình phương nhỏ nhất (Least Squares), trong đó chúng ta tìm kiếm bộ trọng số mà giảm thiểu sai số giữa dự đoán và giá trị thực tế.

Sau khi huấn luyện xong, mô hình hồi quy có thể được sử dụng để dự đoán giá trị mới dựa trên các giá trị đầu vào. Đánh giá mô hình thường được thực hiện bằng cách so sánh giữa dự đoán và giá trị thực tế sử dụng các độ đo như sai số trung bình (Mean Squared Error) hay hệ số xác định (Coefficient of Determination).

Hồi quy là một công cụ mạnh mẽ trong machine learning và được sử dụng rộng rãi trong nhiều lĩnh vực như dự đoán giá cổ phiếu, ước tính giá nhà

## Thuật ngữ:

- Features(inputs), we'll call these x (or xif vectors)
- Training examples, many x(i) for which y(i)is known
  (e.g., many movies for which we know the rating)
- A model, a function that represents the relationship
  between x and y
- A loss/a cost/an objective function, which tells us
  how well our model approximates the training
  examples
- Optimization, a way of finding the parameters of
  our model that minimizes the loss function

## Noise

Định nghĩa: Noise trong Machine Learning là sự không chính xác hoặc sự biến đổi ngẫu nhiên trong dữ liệu mà có thể gây ra sai lệch trong quá trình học và dự đoán. Noise có thể xuất hiện từ nhiều nguồn khác nhau như thiết bị đo lường không chính xác, sai sót trong quá trình thu thập dữ liệu hoặc các quá trình xử lý dữ liệu không chính xác.

### Loại noise:

- Noise thuộc loại Gaussian: Đây là loại noise có phân phối Gaussian, tức là có tính chất ngẫu nhiên và có thể được mô tả bằng một hàm phân phối Gaussian. Loại noise này thường được xem như một yếu tố không mong muốn và ảnh hưởng đáng kể đến quá trình huấn luyện và dự đoán của mô hình.

- Noise chuẩn hoá (Normalization noise): Đây là noise phát sinh do quá trình chuẩn hoá dữ liệu. Khi chuẩn hoá dữ liệu, đặc biệt là trong các phương pháp chuẩn hoá như Z-score, Min-Max, dữ liệu có thể bị biến đổi và tạo ra noise.

- Noise nhiễu (Outlier noise): Đây là noise phát sinh khi có các điểm dữ liệu ngoại lệ, tức là các điểm dữ liệu có giá trị quá lớn hoặc quá nhỏ so với các điểm dữ liệu khác. Noise nhiễu có thể làm sai lệch kết quả huấn luyện và dự đoán.

### Ảnh hưởng của noise:

- Ảnh hưởng đến quá trình huấn luyện: Noise có thể gây ra các sai lệch trong quá trình huấn luyện, khiến mô hình học các mẫu noise thay vì học các mẫu có ý nghĩa thực tế. Điều này dẫn đến hiệu suất dự đoán kém và khả năng tổng quát hóa yếu.

- Ảnh hưởng đến quá trình dự đoán: Noise có thể làm sai lệch kết quả dự đoán của mô hình, đặc biệt là khi mức độ noise lớn và không được kiểm soát. Dự đoán không chính xác có thể dẫn đến các quyết định sai lầm hoặc thông tin không đáng tin cậy.

### Cách xử lý noise:

- Tiền xử lý dữ liệu: Đối với noise thuộc loại Gaussian, có thể sử dụng các phương pháp tiền xử lý như lọc thông tin, loại bỏ các điểm dữ liệu nhiễu hoặc giảm thiểu ảnh hưởng của noise trong quá trình huấn luyện.

- Sử dụng mô hình chống nhiễu: Có thể sử dụng các mô hình chống nhiễu như mô hình chống nhiễu tự động (Autoencoder) hoặc các mô hình chống nhiễu dựa trên thuật toán như hồi quy Ridge (Ridge Regression) hoặc SVM (Support Vector Machines) để giảm thiểu tác động của noise.

- Sử dụng các phương pháp regularization: Các phương pháp regularization như L1 regularization hoặc L2 regularization có thể giúp mô hình chống lại noise và cải thiện khả năng tổng quát hóa.

- Thu thập dữ liệu chính xác: Để giảm thiểu noise, việc thu thập dữ liệu chính xác và kiểm tra dữ liệu trước quá trình huấn luyện là rất quan trọng.

## Type of regression

- Linear regression (Univariate)
- Linear Regression for multiple variables
- Logistic regression

### Linear regression
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/3c8ee232-7c46-419e-9f18-b77544f4d3c1)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/b87125cd-16b5-471f-a9d5-4be045d810fb)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/ab62bf5d-329e-4ba0-9c81-4e1e5e939586)

Loss Function
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/139cf597-5785-431d-928a-f2e6431c87c1)

Gradient Descent
Gradient Descent (GD) là một phương pháp tối ưu trong Machine Learning và optimization để tìm giá trị tối ưu của một hàm mục tiêu. Phương pháp này dựa trên việc điều chỉnh các tham số của mô hình dự đoán dựa trên gradient của hàm mục tiêu.

Cách thức hoạt động của Gradient Descent:
- Khởi tạo các tham số ban đầu: Đầu tiên, các tham số của mô hình được khởi tạo với giá trị ngẫu nhiên hoặc giá trị ban đầu đã cho.
- Tính gradient: Tiếp theo, gradient của hàm mục tiêu được tính toán đối với các tham số hiện tại. Gradient cho biết hướng tăng nhanh nhất của hàm mục tiêu tại vị trí hiện tại.
- Cập nhật tham số: Các tham số của mô hình được cập nhật dựa trên gradient và một hệ số learning rate (tỷ lệ học). Công thức cập nhật thường được sử dụng là: tham số mới = tham số cũ - learning rate * gradient.
- Lặp lại quá trình: Quá trình tính toán gradient và cập nhật tham số được lặp lại cho đến khi đạt được điều kiện dừng, ví dụ như đạt đủ số lần lặp, đạt được độ chính xác mong muốn hoặc hàm mục tiêu không còn thay đổi đáng kể.

Gradient Descent có hai biến thể chính:
- Batch Gradient Descent (BGD): Trong BGD, toàn bộ tập dữ liệu huấn luyện được sử dụng để tính toán gradient và cập nhật tham số. Điều này có thể làm tăng đáng kể thời gian tính toán, nhất là với các tập dữ liệu lớn.

- Stochastic Gradient Descent (SGD): Trong SGD, gradient và cập nhật tham số được tính toán cho từng mẫu dữ liệu huấn luyện. Điều này giúp giảm thời gian tính toán, nhưng có thể gây ra độ dao động và nhiễu trong quá trình tối ưu.

![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c9b1ddc1-90d1-44df-be92-d78fbc254df5)

Gradient Descent cho hàm 1 biến
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/0addbf5d-f07a-4d5a-8a3d-31d119172b7a)

Gradient Descent cho hàm nhiều biến
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/2a61727c-55ef-4052-8b9a-5fa168190d84)

![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/ba41bbdb-d979-49cc-85e8-5fdf178473db)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/48b2bcfa-c4e3-4e0b-8580-abf9ac40f8d6)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/cd1c2efc-8d17-4fc6-95e8-6fcddb7d1be7)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/21de629a-e399-4c9d-be2f-b99343178f89)

Overfitting
Overfitting là hiện tượng xảy ra khi một mô hình học máy quá tập trung vào dữ liệu huấn luyện mà không tổng quát hóa tốt cho dữ liệu mới. Khi một mô hình bị overfitting, nó có khả năng hoạt động tốt trên dữ liệu huấn luyện đã sử dụng để huấn luyện, nhưng không thể dự đoán chính xác trên dữ liệu mới.
Nguyên nhân chính dẫn đến overfitting:
- Sự phức tạp quá mức của mô hình: Nếu mô hình có quá nhiều tham số hoặc quá sức mạnh, nó có thể học những chi tiết không cần thiết từ dữ liệu huấn luyện và dẫn đến việc không tổng quát hóa tốt.
- Kích thước dữ liệu huấn luyện nhỏ: Khi lượng dữ liệu huấn luyện ít, mô hình dễ bị overfitting vì nó có thể học "học thuộc lòng" các mẫu trong tập dữ liệu nhỏ.
- Không cân bằng giữa số lượng mẫu và số lượng đặc trưng: Nếu số lượng đặc trưng (features) lớn hơn đáng kể so với số lượng mẫu, mô hình có thể học những quy tắc quá cụ thể dựa trên các mẫu hiếm hoặc nhiễu trong dữ liệu huấn luyện.
Các dấu hiệu của overfitting:
- Độ chính xác trên tập huấn luyện cao, nhưng độ chính xác trên dữ liệu kiểm tra thấp.
- Sự dao động lớn trong hiệu suất dự đoán trên các tập dữ liệu khác nhau.
- Mô hình có xu hướng "học thuộc lòng" các mẫu trong dữ liệu huấn luyện, như việc phân loại hoặc dự đoán chính xác từng mẫu trong tập huấn luyện.
Cách giảm overfitting:
- Tăng dữ liệu huấn luyện: Bằng cách thu thập thêm dữ liệu huấn luyện, mô hình có thể học từ một tập dữ liệu lớn và tổng quát hóa tốt hơn.
- Giảm sự phức tạp của mô hình: Giảm số lượng tham số hoặc sức mạnh của mô hình có thể giúp tránh overfitting. Có thể sử dụng các phương pháp như regularization (như L1 hoặc L2 regularization) để giảm overfitting.
- Sử dụng kỹ thuật cross-validation: Sử dụng kỹ thuật cross-validation để đánh giá hiệu suất của mô hình trên nhiều tập dữ liệu kiểm tra khác nhau và đảm bảo khả năng tổng quát hóa.
- Áp dụng kỹ thuật dropout: Kỹ thuật dropout trong mạng neural có thể giúp mô hình học từ các đặc trưng đa dạng và tránh overfitting bằng cách tạm thời loại bỏ ngẫu nhiên một số lượng các đơn vị (neuron) trong quá trình huấn luyện.
- Sử dụng mô hình ensemble: Kết hợp nhiều mô hình dự đoán để giảm overfitting và cải thiện khả năng tổng quát hóa.

### Logistic regression
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/a0419dfb-8870-45d6-aa89-645c930ab61c)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c0b968a8-3cf4-406a-93cd-152166de6181)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/562dbc41-dba2-4992-ad63-0f6ff2306587)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/9d08e8d9-2c41-42d1-9965-6960aaa99f62)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/9c3b90cb-50c4-4073-a6fd-eaa6bef0f4e4)

### Model evaluation
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/019fca79-f7d6-46a4-a970-bb25eece0402)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/0f3682ee-d9d6-4fec-aac2-cd67a5998acb)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c2db1787-a00c-4ed2-b25e-4553e063efcf)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/2e4a57ad-a951-4c4f-9155-c839c3ce7fef)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/b8c62a21-9da3-437a-84d6-6418b261a8cb)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/b93acfb8-7dae-4477-b681-fd68cd48e592)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/0178c829-9f31-4d5c-8bf4-725c26111b90)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/1fd0ba74-9fa1-441a-92a4-836659659880)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c6999af2-c252-4e60-9e50-28ef3af94af1)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c2712788-523a-4c5e-bf58-33c0da81d55c)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/4910f8dc-e42f-49b1-a611-8a4824000b3b)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/739e655c-2339-4bfc-8905-37960dad4ae6)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/af39cae4-1ac0-4434-b6d9-5bd80a4b88e9)







