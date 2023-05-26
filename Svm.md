# SVM

Máy vector hỗ trợ (Support Vector Machines - SVM) là một thuật toán trong machine learning (học máy) được sử dụng để phân loại và hồi quy. Được giới thiệu lần đầu tiên bởi Vapnik và Cortes vào năm 1995, SVM đã trở thành một trong những thuật toán phổ biến và mạnh mẽ trong lĩnh vực này.

Trên cơ bản, SVM tạo ra một siêu phẳng (hyperplane) trong không gian đa chiều, nhằm phân tách các điểm dữ liệu thuộc vào các lớp khác nhau. Mục tiêu của SVM là tìm ra siêu phẳng tốt nhất sao cho khoảng cách từ các điểm dữ liệu gần nhất đến siêu phẳng là lớn nhất. Các điểm dữ liệu gần nhất được gọi là vector hỗ trợ (support vectors).

SVM có thể áp dụng cho các bài toán phân loại nhị phân (binary classification) và đa lớp (multi-class classification), cũng như cho bài toán hồi quy (regression). Trong bài toán phân loại, SVM sử dụng một hàm kernel để ánh xạ dữ liệu vào không gian cao chiều (high-dimensional space), nơi việc phân loại dễ dàng hơn. Một số hàm kernel phổ biến được sử dụng là linear kernel, polynomial kernel và radial basis function (RBF) kernel.

Để huấn luyện một mô hình SVM, chúng ta cần xác định các tham số như loại kernel, hệ số regularization và hệ số kernel trong trường hợp sử dụng kernel. Quá trình tìm ra siêu phẳng tối ưu thường được thực hiện bằng các phương pháp tối ưu hóa như Sequential Minimal Optimization (SMO) hoặc Quadratic Programming (QP).

Sau khi huấn luyện xong, mô hình SVM có thể được sử dụng để phân loại dữ liệu mới hoặc dự đoán giá trị đầu ra trong bài toán hồi quy. Đánh giá mô hình SVM thường được thực hiện bằng cách sử dụng các độ đo như độ chính xác (accuracy), độ phủ (recall), độ chính xác dương tính (precision), và F1-score.

SVM là một thuật toán mạnh mẽ và linh hoạt, có khả năng xử lý các bài toán phân loại phức tạp và dữ liệu không tuyến tính. Nó được áp dụng rộng rãi trong nhiều lĩnh vực như xử lý hình ảnh, nhận dạng văn bản

## What is SVM?

Vladimir Vapnik là người đặt các nền tảng
cho SVM trong thời gian làm luận án tiến sĩ tại
Liên Xô những năm 1960

The Support Vector Machine (SVM) là linear
classifier.

Ý tưởng chính của SVM là tìm ra một siêu phẳng (hyperplane) trong không gian đặc trưng (feature space) để phân tách các điểm dữ liệu thuộc vào hai lớp khác nhau. Siêu phẳng này được xác định bằng cách tối đa hoá độ rộng của "margin" (khoảng cách gần nhất từ các điểm dữ liệu gần nhất của hai lớp đến siêu phẳng) và đồng thời đảm bảo các điểm dữ liệu không bị phân loại sai (vi phạm "margin"). Các điểm dữ liệu nằm gần "margin" và có khả năng ảnh hưởng đến vị trí của siêu phẳng được gọi là "support vectors".

Có hai dạng chính của SVM:

- SVM phân loại (SVM for classification): SVM được sử dụng để phân loại các điểm dữ liệu vào các lớp khác nhau. Khi không gian đặc trưng không phân biệt tuyến tính hai lớp, SVM có thể sử dụng các hàm kernel (như kernel tuyến tính, kernel đa thức, kernel Gaussian, kernel sigmoid, v.v.) để ánh xạ không gian đặc trưng vào một không gian khác có thể phân biệt tuyến tính.

- SVM hồi quy (SVM for regression): SVM cũng có thể được sử dụng cho bài toán hồi quy, nơi mục tiêu là dự đoán một giá trị số thực. SVM hồi quy cố gắng tìm ra siêu phẳng mà có độ rộng "margin" lớn nhất để chứa số lượng lớn nhất các điểm dữ liệu mục tiêu trong khoảng cách cho phép.

Một số đặc điểm và ưu điểm của SVM:

- SVM có khả năng xử lý các tập dữ liệu lớn và có số chiều cao.
- SVM tìm kiếm giải pháp tối ưu toàn cục, giúp tránh vấn đề mắc kẹt ở các giá trị cục bộ.
  SVM cũng có khả năng xử lý dữ liệu nhiễu và cực đại hóa độ chính xác.
- Một ưu điểm quan trọng khác của SVM là khả năng sử dụng các hàm kernel để ánh xạ không gian đặc trưng sang một không gian khác, mở rộng khả năng phân loại tuyến tính của SVM đến các không gian phi tuyến.

## Linear Classifier

Linear Classifier (Bộ phân loại tuyến tính) là một thuật toán trong Machine Learning sử dụng một siêu phẳng tuyến tính để phân loại các điểm dữ liệu vào các lớp khác nhau. Siêu phẳng tuyến tính là một đường thẳng, mặt phẳng hoặc siêu phẳng trong không gian đặc trưng mà tách rời các điểm dữ liệu của các lớp khác nhau.

Ý tưởng chính của Linear Classifier là sử dụng các hàm tuyến tính để xác định vị trí của siêu phẳng tuyến tính, dựa trên giá trị của các đặc trưng đầu vào. Các đặc trưng của mỗi điểm dữ liệu được biểu diễn dưới dạng vector, và các trọng số tương ứng với từng đặc trưng được xác định để tạo ra phương trình của siêu phẳng.

Công thức chung cho một bộ phân loại tuyến tính đơn giản có thể được biểu diễn như sau:
y = w1x1 + w2x2 + ... + wnxn + b

Trong đó:

- y là đầu ra dự đoán.
- x1, x2, ..., xn là các đặc trưng đầu vào.
- w1, w2, ..., wn là các trọng số tương ứng với từng đặc trưng.
- b là hệ số điều chỉnh (bias).

Việc tìm các trọng số w1, w2, ..., wn và hệ số điều chỉnh b được thực hiện thông qua quá trình huấn luyện, trong đó mô hình cố gắng tìm các giá trị này để phân loại đúng các điểm dữ liệu trong tập huấn luyện. Các thuật toán huấn luyện như Gradient Descent, Stochastic Gradient Descent, hoặc các phương pháp tối ưu khác có thể được sử dụng để tìm giá trị tối ưu của các tham số.

Linear Classifier thường được sử dụng cho các bài toán phân loại nhị phân, trong đó chỉ có hai lớp dữ liệu cần được phân loại. Tuy nhiên, nó cũng có thể được mở rộng để xử lý các bài toán phân loại đa lớp bằng các phương pháp như One-vs-Rest (một lớp so với còn lại) hoặc One-vs-One (một lớp so với lớp khác).

Margin (khoảng cách) của bộ phân loại (classifier) là độ rộng của "vùng an toàn" giữa siêu phẳng tuyến tính và các điểm dữ liệu gần nhất của các lớp khác nhau. Nó đo lường sự tách rời giữa các lớp và cho phép đánh giá độ tổng quát hóa của bộ phân loại.

Đối với một bộ phân loại tuyến tính, margin được tính bằng khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất của các lớp khác nhau. Margin lớn hơn đồng nghĩa với việc siêu phẳng tách rời các lớp một cách rõ ràng hơn và có khả năng tổng quát hóa tốt hơn.

Công thức tính toán margin trong trường hợp đơn giản của bộ phân loại tuyến tính (với hai lớp) có thể được biểu diễn như sau:
margin = (2 / ||w||)

Trong đó:

- w là vector trọng số của siêu phẳng tuyến tính.
- ||w|| là độ dài (norm) của vector trọng số.

Công thức này giả định rằng siêu phẳng được chuẩn hóa sao cho ||w|| = 1. Khi cần thiết, các phương pháp tối ưu hóa có thể được sử dụng để tìm siêu phẳng tốt nhất và tối đa hóa margin.

Margin lớn cho thấy rằng bộ phân loại có độ tổng quát hóa cao và có khả năng tách rời các lớp dữ liệu tốt hơn. Khi margin nhỏ, tức là siêu phẳng gần các điểm dữ liệu của các lớp, có khả năng bộ phân loại bị overfitting và không tổng quát hóa tốt cho dữ liệu mới.

Tối đa hóa margin là một mục tiêu quan trọng trong SVM (Support Vector Machine) và một số bộ phân loại tuyến tính khác. Quá trình huấn luyện của các thuật toán này thường tìm kiếm siêu phẳng có margin lớn nhất để đảm bảo khả năng tổng quát hóa tốt và giảm thiểu overfitting.

Objective function

Objective function của SVM thường bao gồm hai thành phần chính: thành phần regularization (điều chuẩn) và thành phần loss (mất mát). Để hiểu rõ hơn, ta sẽ xem xét hai dạng phổ biến của SVM: SVM phân loại nhị phân (SVM for classification) và SVM hồi quy (SVM for regression).

1. Objective function của SVM phân loại nhị phân:

Trong SVM phân loại nhị phân, mục tiêu là tìm siêu phẳng tối ưu để tách rời các điểm dữ liệu của hai lớp. Objective function của SVM phân loại nhị phân bao gồm hai thành phần chính:

- Thành phần regularization (điều chuẩn): Điều chuẩn được sử dụng để giới hạn độ phức tạp của mô hình và tránh overfitting. Thông thường, SVM sử dụng L2 regularization, được biểu diễn bằng ||w||^2, trong đó w là vector trọng số.

- Thành phần loss (mất mát): Thành phần loss đo lường độ sai lệch của các điểm dữ liệu so với siêu phẳng tối ưu. Một hàm loss phổ biến trong SVM là hinge loss, được biểu diễn bằng max(0, 1 - y\*f(x)), trong đó y là nhãn lớp (-1 hoặc 1), f(x) là giá trị đầu ra của siêu phẳng cho điểm dữ liệu x.

Objective function của SVM phân loại nhị phân được biểu diễn như sau:
minimize: (1/2) _ ||w||^2 + C _ Σ[max(0, 1 - y*f(x))]

Trong đó:

- (1/2) \* ||w||^2 là thành phần regularization.
- C là hệ số điều chỉnh (hyperparameter) để điều chỉnh sự cân bằng giữa thành phần regularization và thành phần loss.
- Σ[max(0, 1 - y*f(x))] là tổng các giá trị của hàm hinge loss cho các điểm dữ liệu.

2. Objective function của SVM hồi quy:

Trong SVM hồi quy, mục tiêu là tìm siêu phẳng tối ưu để dự đoán giá trị liên tục. Objective function của SVM hồi quy cũng bao gồm thành phần regularization và thành phần loss, tuy nhiên, hàm loss được sử dụng khác với SVM phân loại nhị phân.

Objective function của SVM hồi quy được biểu diễn như sau:
minimize: (1/2) _ ||w||^2 + C _ Σ[loss(y - f(x))]

Trong đó:

- (1/2) \* ||w||^2 là thành phần regularization.
- C là hệ số điều chỉnh (hyperparameter) để điều chỉnh sự cân bằng giữa thành phần regularization và thành phần loss.
- Σ[loss(y - f(x))] là tổng các giá trị của hàm loss (ví dụ như hàm mất mát tuyến tính hoặc hàm mất mát epsilon-insensitive) cho các điểm dữ liệu.
