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

Lagrange multipliers

Là một phương pháp trong tối ưu hóa được sử dụng để tìm các giá trị tối ưu của một hàm mục tiêu trong những ràng buộc. Phương pháp này được đặt tên theo nhà toán học người Ý Joseph-Louis Lagrange.
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c983b195-5163-4206-8aa4-2175b35eca28)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/9fc8b997-3e51-44a8-8bfc-f6a657c6d99e)

Soft margin SVM
Soft Margin SVM (Support Vector Machine) là một biến thể của SVM được sử dụng để xây dựng bộ phân loại có khả năng chấp nhận sự mất mát (vi phạm ràng buộc) trong quá trình tạo ra siêu phẳng tối ưu. Phương pháp này giúp SVM xử lý hiệu quả những tập dữ liệu có sự chồng chéo hoặc nhiễu.

Trong SVM truyền thống, mục tiêu là tìm một siêu phẳng tuyến tính tối ưu để tách rời hai lớp dữ liệu một cách tuyệt đối. Tuy nhiên, trong một số trường hợp, không tồn tại một siêu phẳng hoàn hảo để hoàn toàn tách rời các điểm dữ liệu. Đây là lúc mà Soft Margin SVM xuất hiện.

Soft Margin SVM cho phép một số lượng nhất định các điểm dữ liệu rơi vào vùng không đúng lớp (vi phạm ràng buộc). Thay vì đòi hỏi mọi điểm dữ liệu phải nằm chính xác hai bên của siêu phẳng, Soft Margin SVM cho phép một vài điểm dữ liệu nằm trong vùng an toàn (margin) của siêu phẳng hoặc thậm chí nằm sai lớp nhưng vẫn cố gắng giảm thiểu tổng số vi phạm ràng buộc.

Để điều chỉnh sự đồng thuận giữa việc tối ưu hóa margin và vi phạm ràng buộc, Soft Margin SVM sử dụng tham số C, được gọi là hằng số penalty. Tham số này quyết định mức độ chấp nhận cho vi phạm ràng buộc. Giá trị C càng lớn, SVM càng cố gắng tối đa hóa margin và chấp nhận ít vi phạm ràng buộc. Ngược lại, giá trị C càng nhỏ, SVM càng chấp nhận nhiều vi phạm ràng buộc để có margin lớn hơn.

Quá trình tối ưu hóa của Soft Margin SVM có thể được thực hiện bằng phương pháp tối ưu hóa bậc hai như phương pháp Lagrange Multipliers và SMO (Sequential Minimal Optimization).

Kernel trick 
Kernel Trick là một kỹ thuật được sử dụng trong Support Vector Machine (SVM) để mở rộng khả năng phân loại của SVM từ không gian đặc trưng ban đầu lên một không gian đặc trưng cao hơn, mà có thể giúp SVM xử lý hiệu quả các bài toán phân loại phi tuyến.

Trong SVM, mục tiêu là tìm một siêu phẳng tuyến tính tối ưu để tách rời các điểm dữ liệu của các lớp khác nhau. Tuy nhiên, nếu dữ liệu không thể tách rời hoàn toàn bằng một siêu phẳng tuyến tính, SVM sẽ không hoạt động hiệu quả.

Kernel Trick giúp giải quyết vấn đề này bằng cách ánh xạ dữ liệu từ không gian đặc trưng ban đầu vào một không gian đặc trưng cao hơn thông qua một hàm ánh xạ phi tuyến (nonlinear mapping). Hàm ánh xạ này được gọi là kernel function.

Kernel function tính toán một đại lượng gọi là độ tương tự (similarity measure) giữa hai điểm dữ liệu trong không gian đặc trưng cao hơn. Khi sử dụng kernel function, SVM có thể xử lý dữ liệu phi tuyến mà không cần thực hiện việc tăng số chiều của dữ liệu. Điều này giúp giảm độ phức tạp tính toán và tiết kiệm bộ nhớ.

Một số kernel function phổ biến được sử dụng trong SVM bao gồm:
- Linear Kernel: K(x, y) = x^T * y
- Polynomial Kernel: K(x, y) = (x^T * y + c)^d
- Gaussian (RBF) Kernel: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

Multi-class SVM
Multi-Class SVM (Support Vector Machine) là một biến thể của SVM được sử dụng để xây dựng mô hình phân loại cho các bài toán có nhiều hơn hai lớp dữ liệu. Trong SVM truyền thống, chỉ có thể xử lý bài toán phân loại nhị phân, trong đó dữ liệu được chia thành hai lớp. Tuy nhiên, với Multi-Class SVM, chúng ta có thể xử lý bài toán phân loại đa lớp, trong đó dữ liệu được chia thành ba lớp trở lên.

Có hai phương pháp chính để xây dựng một Multi-Class SVM: One-vs-One và One-vs-All (One-vs-Rest).
- One-vs-One (OvO): Trong phương pháp này, mỗi cặp lớp dữ liệu được chọn để tạo thành một bài toán phân loại nhị phân riêng biệt. Ví dụ, nếu chúng ta có k lớp dữ liệu, thì chúng ta sẽ tạo ra k(k-1)/2 bài toán phân loại nhị phân. Mỗi bài toán phân loại sẽ xác định xem một điểm dữ liệu thuộc lớp nào trong hai lớp được chọn. Kết quả của tất cả các bài toán phân loại sẽ được tổng hợp để quyết định lớp cuối cùng của điểm dữ liệu.

- One-vs-All (OvA hoặc OvR): Trong phương pháp này, mỗi lớp dữ liệu được chọn để tạo thành một bài toán phân loại nhị phân riêng biệt. Với mỗi bài toán phân loại, lớp được chọn được coi là lớp dương và các lớp khác được coi là lớp âm. Kết quả của tất cả các bài toán phân loại sẽ được đưa ra và lớp có điểm cao nhất sẽ được chọn là lớp cuối cùng của điểm dữ liệu.

Tổng kết:
SVMs find optimal linear separator
Kernel Trick giúp SVM học các bề mặt quyết định phi tuyến.
Ưu điểm của SVM:
- Hiệu suất lý thuyết và thực nghiệm tốt.
- Hỗ trợ nhiều loại kernel khác nhau.
Nhược điểm của SVM:
- Tốn thời gian huấn luyện và dự đoán đối với các tập dữ liệu lớn (tuy nhiên, vẫn nhanh so với một số phương pháp khác).
- Lần chọn kernel (và điều chỉnh các tham số của nó).

Classification
Classification là quá trình phân loại các điểm dữ liệu vào các lớp hay nhãn khác nhau dựa trên các đặc trưng hoặc thuộc tính của chúng. Đây là một bài toán quan trọng trong Machine Learning và có nhiều phương pháp khác nhau để thực hiện quá trình phân loại.

Một bài toán phân loại thường bao gồm hai phần chính: dữ liệu huấn luyện và một mô hình phân loại. Trong quá trình huấn luyện, mô hình phân loại được huấn luyện trên một tập dữ liệu được gán nhãn trước, trong đó mỗi điểm dữ liệu có một nhãn hoặc lớp xác định. Mô hình học các mẫu và quy luật từ dữ liệu huấn luyện để phân loại các điểm dữ liệu mới.

Phân chia dựa trên Base Classifiers và Ensemble Classifiers là hai cách tiếp cận khác nhau trong việc xây dựng các mô hình phân loại.
Base Classifiers (Bộ phân loại cơ bản): Các Base Classifiers là các mô hình phân loại đơn lẻ độc lập với nhau. Mỗi bộ phân loại cơ bản được huấn luyện trên một tập dữ liệu con hoặc trên toàn bộ tập dữ liệu gốc. Các bộ phân loại cơ bản này có thể là các thuật toán phân loại đơn giản như Logistic Regression, Decision Trees, Support Vector Machines, Naive Bayes, Neural Networks và nhiều thuật toán khác.
- Hồi quy Logistic (Logistic Regression): Đây là một mô hình tuyến tính được sử dụng cho phân loại nhị phân. Mô hình này mô phỏng xác suất của một điểm dữ liệu thuộc về một lớp cụ thể bằng cách sử dụng hàm logistic.

- Cây quyết định (Decision Trees): Đây là các mô hình phân loại có cấu trúc phân cấp dựa trên một chuỗi các quy tắc. Mỗi nút trong cây biểu thị một kiểm tra trên một đặc trưng, và mỗi nút lá biểu thị một nhãn lớp.

- Rừng ngẫu nhiên (Random Forest): Đây là một phương pháp tổ hợp kết hợp nhiều cây quyết định. Mỗi cây được huấn luyện trên một tập con ngẫu nhiên của dữ liệu, và dự đoán cuối cùng được thực hiện bằng cách kết hợp dự đoán của từng cây.

- Máy vector hỗ trợ (Support Vector Machines - SVM): SVM tìm một siêu mặt phẳng tối ưu chia các điểm dữ liệu thành các lớp khác nhau sao cho độ rộng biên lớn nhất. SVM có thể xử lý cả bài toán phân loại tuyến tính và phi tuyến tính bằng cách sử dụng các kernel khác nhau.

- Naive Bayes: Đây là một bộ phân loại xác suất dựa trên định lý Bayes và giả định độc lập giữa các đặc trưng. Nó tính toán xác suất hậu nghiệm của mỗi lớp dựa trên các đặc trưng đầu vào.

- K-Nearest Neighbors (KNN): Đây là một thuật toán học lười lập dựa trên việc phân loại một điểm dữ liệu dựa trên các lớp của k hàng xóm gần nhất trong không gian đặc trưng.

- Mạng neural (Neural Networks): Đây là một lớp mô hình được lấy cảm hứng từ não người, gồm các nút hoặc "neuron" được kết nối với nhau. Mạng neural có thể xử lý các mẫu và mối quan hệ phức tạp trong dữ liệu, và được sử dụng rộng rãi trong các bài toán phân loại.

Ensemble Classifiers (Bộ phân loại tổ hợp): Ensemble Classifiers kết hợp các Base Classifiers lại với nhau để tạo thành một mô hình phân loại mạnh hơn. Các phương pháp Ensemble phổ biến bao gồm:
- Voting: Các Base Classifiers đưa ra dự đoán riêng lẻ và dự đoán cuối cùng được xác định dựa trên đa số phiếu bầu từ các bộ phân loại cơ bản. Voting có thể được thực hiện theo hai hình thức: Hard Voting (phiếu bầu cứng) và Soft Voting (phiếu bầu mềm).

- Bagging: Các Base Classifiers được huấn luyện trên các tập dữ liệu con được chọn ngẫu nhiên từ tập dữ liệu huấn luyện gốc. Kết quả dự đoán cuối cùng được tính bằng cách kết hợp các dự đoán từ các bộ phân loại cơ bản, thường là bằng cách lấy trung bình hoặc phiếu bầu.

- Boosting: Các Base Classifiers được huấn luyện tuần tự và tập trung vào việc sửa các sai lầm của các bộ phân loại trước đó. Mỗi bộ phân loại sau đó được tạo ra để tập trung vào các điểm dữ liệu mà các bộ phân loại trước đó dự đoán sai. Kết quả dự đoán cuối cùng là sự kết hợp của các dự đoán từ tất cả các bộ phân loại.

- Stacking: Các Base Classifiers được huấn luyện trên toàn bộ tập dữ liệu huấn luyện và đưa ra dự đoán riêng lẻ. Các dự đoán này sau đó được sử dụng làm đầu vào cho một mô hình Meta-Classifier (thường là một mô hình tuyến tính) để đưa ra dự đoán cuối cùng.

Bayes Classification methods
Các phương pháp phân loại Bayes là một họ các kỹ thuật phân loại dựa trên lý thuyết xác suất Bayes. Những phương pháp này sử dụng các mô hình xác suất để đưa ra dự đoán và gán nhãn lớp cho dữ liệu đầu vào. Dưới đây là một số phương pháp phân loại Bayes phổ biến:
- Bộ phân loại Naive Bayes: Naive Bayes là một thuật toán phân loại đơn giản và nhanh chóng. Nó giả định rằng tất cả các đặc trưng là độc lập với nhau khi đã biết nhãn lớp, do đó có tên "naive". Nó tính toán xác suất hậu nghiệm của mỗi lớp dựa trên các đặc trưng đầu vào bằng cách sử dụng định lý Bayes và chọn lớp có xác suất cao nhất.
- Mạng tin cậy Bayes: Mạng tin cậy Bayes, còn được gọi là mạng Bayes hoặc mô hình đồ thị, biểu diễn các phụ thuộc giữa các biến bằng cách sử dụng một đồ thị có hướng không có chu trình. Các mạng này được sử dụng để mô hình hóa mối quan hệ phức tạp và không chắc chắn trong dữ liệu. Chúng có thể được sử dụng cho cả các nhiệm vụ phân loại và suy luận.
- Hồi quy logistic Bayes: Hồi quy logistic Bayes mở rộng hồi quy logistic truyền thống bằng cách tích hợp phân phối tiên nghiệm vào các tham số của mô hình. Điều này cung cấp một cách để tích hợp tri thức tiên nghiệm hoặc niềm tin về các tham số vào quá trình phân loại. Hồi quy logistic Bayes có thể xử lý cả các bài toán phân loại nhị phân và phân loại đa lớp.
- Cây quyết định Bayes: Cây quyết định Bayes kết hợp nguyên tắc của cây quyết định với suy luận Bayes. Chúng sử dụng cấu trúc cây quyết định để phân chia dữ liệu theo các giá trị đặc trưng và tính toán xác suất hậu nghiệm của các nhãn lớp tại mỗi nút. Phương pháp này cho phép mô hình hóa không chắc chắn một cách rõ ràng và có thể hữu ích khi làm việc với dữ liệu nhỏ hoặc nhiễu.
- Naive Bayes Gaussian: Naive Bayes Gaussian là một biến thể của bộ phân loại Naive Bayes giả định rằng các đặc trưng tuân theo phân phối Gaussian (chuẩn). Nó hoạt động tốt cho các đặc trưng liên tục hoặc có giá trị thực. Nó ước tính trung bình và phương sai của mỗi đặc trưng cho mỗi lớp và sử dụng chúng để tính toán xác suất và xác suất hậu nghiệm.
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/db9ff8ef-8e11-4578-9137-2f7044e737d1)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/bf5f4c77-c602-4071-8eae-86c686d49280)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/e35725d7-525f-4f33-bc0c-c41334610bb2)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/9645e5ba-4375-4222-bb77-29f54f57656b)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/5e944cf6-9351-4b95-acd2-90f0c7eee377)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/bfd941eb-f8dd-4b8e-9f58-1c0952e0f146)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/f3b4e09e-0822-4aca-8c7d-b7afd399253b)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/bf620c0e-e9df-4bdd-b27d-7aa8acc6255b)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/982eae53-2e16-49d7-acda-7f109c89aca0)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/a50545a9-13d8-4ef5-8b16-1dbbbdc88b81)




