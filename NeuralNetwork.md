# Neural networks

Mạng neural (Neural networks) là một thuật toán trong lĩnh vực machine learning (học máy) được lấy cảm hứng từ cách hoạt động của hệ thống thần kinh sinh học. Mạng neural được sử dụng để học và nhận biết các mô hình phức tạp từ dữ liệu đầu vào.

Một mạng neural bao gồm các "neuron" nhân tạo được tổ chức thành các lớp (layers). Các lớp này kết nối với nhau thông qua các trọng số, và thông tin được truyền từ lớp đầu vào (input layer) qua các lớp ẩn (hidden layers) đến lớp đầu ra (output layer). Mỗi neuron tính toán đầu ra dựa trên tổ hợp tuyến tính của đầu vào và hàm kích hoạt phi tuyến (non-linear activation function).

Quá trình huấn luyện mạng neural bao gồm việc điều chỉnh các trọng số của mạng để tối thiểu hóa sai số giữa kết quả dự đoán và kết quả thực tế. Các phương pháp tối ưu hóa như Gradient Descent và Backpropagation thường được sử dụng để cập nhật trọng số.

Mạng neural có thể có kiến trúc đơn giản với chỉ một lớp ẩn hoặc kiến trúc phức tạp với nhiều lớp ẩn. Một số kiến trúc mạng neural phổ biến bao gồm mạng neural truyền thẳng (feedforward neural network), mạng neural hồi quy (recurrent neural network), và mạng neural chập (convolutional neural network).

Mạng neural có khả năng học và nhận biết các mô hình phức tạp, như phân loại hình ảnh, nhận dạng giọng nói, dịch máy và nhiều bài toán khác. Đánh giá hiệu suất của mạng neural thường được thực hiện bằng cách sử dụng các độ đo như độ chính xác (accuracy), độ mất mát (loss) và ma trận lỗi (confusion matrix).

Mạng neural đã trở thành một công cụ quan trọng trong lĩnh vực machine learning, và sự phát triển của các kiến trúc và thuật toán mới đã đóng góp đáng kể vào sự tiến bộ của lĩnh vực này.

## Linear classifier

Một bộ phân loại tuyến tính là một loại bộ phân loại sử dụng ranh giới quyết định tuyến tính để phân tách các lớp khác nhau trong một bài toán phân loại. Đây là một thuật toán phân loại đơn giản và phổ biến, hoạt động tốt khi các lớp có thể được phân tách bằng một đường thẳng hoặc mặt phẳng tuyến tính trong các không gian chiều cao.

Một bộ phân loại tuyến tính là một loại bộ phân loại sử dụng ranh giới quyết định tuyến tính để phân tách các lớp khác nhau trong một bài toán phân loại. Đây là một thuật toán phân loại đơn giản và phổ biến, hoạt động tốt khi các lớp có thể được phân tách bằng một đường thẳng hoặc mặt phẳng tuyến tính trong các không gian chiều cao.

Một số bộ phân loại tuyến tính phổ biến bao gồm:

- Hồi quy Logistic (Logistic Regression): Đây là một bộ phân loại tuyến tính phổ biến, mô hình hóa mối quan hệ giữa các đặc trưng đầu vào và xác suất lớp bằng các hàm logistic hoặc softmax. Nó hoạt động tốt cho các bài toán phân loại nhị phân cũng như phân loại đa lớp.

- Máy hỗ trợ Vector (Support Vector Machines - SVM): SVM là một bộ phân loại tuyến tính mạnh mẽ, tìm một mặt phẳng tuyến tính tối ưu để tối đa hóa biên giữa các lớp khác nhau. Nó cũng có thể xử lý các bài toán phân loại phi tuyến bằng cách sử dụng kỹ thuật kernel để ánh xạ dữ liệu vào không gian chiều cao hơn.

- Phân tích Đa thức (Linear Discriminant Analysis - LDA): LDA là một kỹ thuật giảm chiều dữ liệu có thể được sử dụng như một bộ phân loại tuyến tính. Nó tìm một phép chiếu của dữ liệu tối đa hóa tính tách lớp.

## Perceptron

Perceptron là một thuật toán máy học thuộc loại học tăng cường (supervised learning) và được sử dụng trong bài toán phân loại nhị phân (binary classification). Nó được đề xuất bởi Frank Rosenblatt vào những năm 1950 và là một trong những thuật toán đơn giản nhất trong lĩnh vực học máy.

Perceptron là một mô hình tuyến tính đơn giản của một neuron nhân tạo (artificial neuron). Nó nhận đầu vào là một vectơ đặc trưng và đưa ra dự đoán về lớp của dữ liệu. Công việc của Perceptron là học một hàm quyết định (decision function) để phân tách các điểm dữ liệu thuộc hai lớp khác nhau.

Thuật toán Perceptron hoạt động dựa trên việc cập nhật trọng số (weights) của các đặc trưng để điều chỉnh mô hình. Ban đầu, các trọng số được khởi tạo ngẫu nhiên hoặc bằng 0. Sau đó, mô hình được huấn luyện trên tập dữ liệu huấn luyện thông qua việc tính toán đầu ra của Perceptron và so sánh với nhãn đúng của dữ liệu. Dựa trên sai số dự đoán, trọng số được điều chỉnh bằng cách cập nhật theo hướng giúp cải thiện dự đoán tiếp theo.

Thuật toán Perceptron có thể học được một đường ranh giới quyết định tuyến tính đơn giản, nhưng không thể xử lý các bài toán phân loại phi tuyến. Tuy nhiên, khi kết hợp với các kỹ thuật như kỹ thuật kernel, Perceptron có thể mở rộng để giải quyết các bài toán phân loại phi tuyến phức tạp hơn.

### Perceptron Learning Algorithm

Bước 1: Khởi tạo ngẫu nhiên w (trọng số), t=0.

Bước 2: Xác định DM - tập các điểm dữ liệu được
phân loại sai. Nếu DM rỗng, DỪNG. Ngược lại, tiếp tục Bước 3.

Bước 3: Cập nhật w cho mỗi điểm dữ liệu x(i) thuộc DM.

Bước 4: t=t+1, quay lại Bước 2.

Trong thuật toán trên, ta bắt đầu bằng việc khởi tạo ngẫu nhiên các trọng số w và đặt t=0. Tiếp theo, ta xác định tập DM chứa các điểm dữ liệu bị phân loại sai. Nếu DM không còn chứa điểm dữ liệu nào, tức là không còn lỗi phân loại, ta dừng thuật toán. Ngược lại, ta tiếp tục bằng việc cập nhật trọng số w cho mỗi điểm dữ liệu trong DM để điều chỉnh mô hình. Sau đó, ta tăng t lên 1 và quay lại Bước 2 để tiếp tục quá trình cho đến khi không còn điểm dữ liệu bị phân loại sai hoặc đạt đến một số lần lặp tối đa.

### Pocket Algorithm

Khởi tạo w_pocket ngẫu nhiên

Đối với mỗi lần lặp huấn luyện t:

- Thực hiện thuật toán Perceptron Learning Algorithm (PLA) để có được vector trọng số w_t đã cập nhật.
- Đánh giá w_t và w_pocket (dựa trên số điểm phân loại sai).
- Nếu w_t tốt hơn w_pocket, thay thế w_pocket bằng w_t.

Trả về w_pocket khi quá trình huấn luyện kết thúc.

### Summary

Ưu điểm của Perceptron:

- Perceptron có thể triển khai các cổng logic như AND, OR hoặc NAND.
- Thuật toán Perceptron đơn giản và dễ hiểu.
  Nó là một trong những thuật toán học máy đầu tiên và tạo nền tảng cho nhiều phương pháp học máy khác.

Nhược điểm của Perceptron:

- Perceptron chỉ có thể học các vấn đề có thể phân tách tuyến tính như vấn đề boolean AND.
- Đối với các vấn đề phi tuyến như vấn đề boolean XOR, Perceptron không hoạt động.
- Nếu dữ liệu không phân tách tuyến tính, Perceptron sẽ không thể hội tụ và cho kết quả phân loại chính xác.
- Perceptron không xử lý được các vấn đề phức tạp có nhiều lớp đặc trưng hoặc không gian phức tạp.

## What is Neural network?

Mạng neural, còn được gọi là mạng neural nhân tạo (ANN) hoặc đơn giản là mạng neural, là một mô hình tính toán được lấy cảm hứng từ cấu trúc và hoạt động của não bộ sinh học. Nó bao gồm các nút được kết nối với nhau, được gọi là nơ-ron nhân tạo hoặc "nút", được tổ chức thành các lớp. Mỗi nơ-ron nhận đầu vào, thực hiện một phép tính sử dụng trọng số và sai số, và tạo ra đầu ra.

Mạng neural thường bao gồm ba loại lớp chính: lớp đầu vào, các lớp ẩn và lớp đầu ra. Lớp đầu vào nhận dữ liệu đầu vào, các lớp ẩn xử lý thông tin và lớp đầu ra tạo ra đầu ra cuối cùng hoặc dự đoán.

Các kết nối giữa các nơ-ron trong các lớp khác nhau được biểu diễn bằng trọng số. Các trọng số này xác định sức mạnh của các kết nối và được điều chỉnh trong quá trình huấn luyện để tối ưu hóa hiệu suất của mạng. Sai số là các tham số bổ sung giúp điều chỉnh hàm kích hoạt của mỗi nơ-ron, tạo sự linh hoạt cho mô hình.

Mạng neural được huấn luyện bằng một quá trình gọi là lan truyền ngược (backpropagation), trong đó dữ liệu đầu vào được đưa qua mạng, so sánh đầu ra dự đoán với đầu ra mong muốn, và điều chỉnh trọng số và sai số lặp lại để giảm thiểu sai số. Quá trình huấn luyện này nhằm tối ưu hóa khả năng của mạng để học và tổng quát hóa các mẫu từ dữ liệu.

Mạng neural đã trở nên ngày càng phổ biến nhờ khả năng học các mẫu phức tạp và đưa ra dự đoán chính xác trong các lĩnh vực khác nhau, bao gồm nhận dạng hình ảnh và giọng nói, xử lý ngôn ngữ tự nhiên và phân tích dữ liệu. Chúng cũng là những khối xây dựng của học sâu (deep learning), một lĩnh vực con của học máy tập trung vào việc huấn luyện mạng neural với nhiều lớp ẩn.

### Brief history of neural networks

1943: Mô hình Nơ-ron McCulloch-Pitts - Warren McCulloch và Walter Pitts giới thiệu một mô hình toán học của nơ-ron nhân tạo, đặt nền tảng cho nghiên cứu mạng neural.

Thập kỷ 1950-1960: Perceptron - Frank Rosenblatt phát triển perceptron, một mạng neural một lớp có khả năng học và thực hiện phân loại nhị phân. Perceptron là một thành công đầu tiên trong nghiên cứu mạng neural nhưng có giới hạn trong việc giải quyết các vấn đề phức tạp.

Những năm 1980: Thuật toán Backpropagation - Thuật toán backpropagation, được phát triển bởi Paul Werbos và sau đó trở nên phổ biến do David Rumelhart, Geoffrey Hinton và Ronald Williams, đã cách mạng hóa quá trình huấn luyện mạng neural. Backpropagation cho phép tính toán hiệu quả các cập nhật trọng số, cho phép huấn luyện các mạng neural nhiều lớp.

Những năm 1980-1990: Sự phục hưng của Mạng Neural - Mạng neural đã trải qua một đợt phát triển mạnh mẽ và sự quan tâm trong thời kỳ này. Các nhà nghiên cứu khám phá các kiến trúc khác nhau, hàm kích hoạt và thuật toán học, dẫn đến các tiến bộ trong nhận dạng mẫu, nhận dạng giọng nói và xử lý hình ảnh.

Những năm 1990-2000: Ưu thế của Máy vector hỗ trợ (SVM) - Máy vector hỗ trợ trở nên phổ biến nhờ vào nền tảng lý thuyết mạnh mẽ và hiệu suất ấn tượng trên nhiều nhiệm vụ. Mạng neural trở nên ít được chú trọng khi SVM thu hút sự chú ý trong cộng đồng học máy.

Những năm 2000-nay: Cuộc cách mạng Deep Learning - Vào đầu những năm 2000, các kỹ thuật Deep Learning bắt đầu thu hút sự quan tâm, được thúc đẩy bởi sự tiến bộ về sức mạnh tính toán và

## Network structures

Các network structures phổ biến trong mạng neural bao gồm:

- Feedforward Neural Networks: Đây là loại mạng neural cơ bản, trong đó thông tin chỉ di chuyển theo một hướng từ đầu vào đến đầu ra mà không có các kết nối phản hồi (feedback connections). Feedforward neural networks thường bao gồm các lớp input (đầu vào), các lớp ẩn (hidden layers), và lớp output (đầu ra).

- Convolutional Neural Networks (CNN): CNN là một loại mạng neural được thiết kế đặc biệt cho xử lý hình ảnh và dữ liệu không gian. CNN sử dụng các lớp convolution (lớp tích chập) để trích xuất các đặc trưng từ hình ảnh và các lớp pooling (lớp gộp) để giảm kích thước của đặc trưng. CNN thường được sử dụng trong các nhiệm vụ như nhận dạng đối tượng và phân loại ảnh.

- Recurrent Neural Networks (RNN): RNN là một loại mạng neural có khả năng xử lý dữ liệu tuần tự hoặc dữ liệu có mối quan hệ thời gian. RNN sử dụng các đơn vị đặc biệt được gọi là mạng nơ-ron đệm (memory cells) để lưu trữ thông tin từ các bước trước đó và chia sẻ nó qua các bước tiếp theo. RNN thường được sử dụng trong các nhiệm vụ như dịch máy, xử lý ngôn ngữ tự nhiên và dự đoán chuỗi thời gian.

- Long Short-Term Memory (LSTM) Networks: LSTM là một biến thể của RNN, được thiết kế để giải quyết vấn đề của việc truyền thông tin trong các chuỗi dài. LSTM sử dụng các cơ chế đặc biệt để lưu trữ và quyết định thông tin quan trọng để truyền qua các bước tiếp theo. LSTM thường được sử dụng trong các nhiệm vụ yêu cầu xử lý thông tin dựa trên ngữ cảnh dài, chẳng hạn như nhận dạng giọng nói và dịch máy.

## Multilayer Perceptron

### What is Multilayer Perceptron?

Multi-layer perceptron (MLP) là một kiến trúc mạng neural được sử dụng rộng rãi trong machine learning. Nó là một dạng mạng neural feedforward, có nhiều lớp ẩn (hidden layers) giữa lớp input và lớp output. MLP có khả năng học và biểu diễn các hàm phi tuyến tính phức tạp thông qua việc kết hợp các tác động phi tuyến tính từ các lớp ẩn.

Mỗi lớp trong MLP bao gồm một số lượng nơ-ron (neuron) được kết nối với lớp trước và lớp sau bằng các trọng số. Các nơ-ron trong một lớp tính tổng các đầu vào từ lớp trước và áp dụng một hàm kích hoạt phi tuyến tính để tính toán đầu ra của nó. Các lớp ẩn giữa lớp input và lớp output có thể có nhiều nơ-ron và có thể sử dụng các hàm kích hoạt khác nhau.

Quá trình huấn luyện MLP thường bao gồm việc tối ưu hóa các trọng số trong mạng neural thông qua thuật toán gradient descent và backpropagation. Backpropagation là một phương pháp tính toán độ lỗi và lan truyền ngược thông tin lỗi từ lớp output đến lớp input để điều chỉnh các trọng số. Khi được huấn luyện đủ, MLP có khả năng xử lý các nhiệm vụ phức tạp như phân loại và dự đoán.

### Why MLPs

MLPs (Multi-layer perceptrons) là một trong những kiến trúc mạng neural phổ biến và được sử dụng rộng rãi trong machine learning và deep learning. Có một số lý do chính vì sao MLPs được ưa chuộng:

- Khả năng giải quyết các vấn đề phi tuyến: MLPs có khả năng học và biểu diễn các hàm phi tuyến, cho phép chúng giải quyết nhiều loại bài toán phức tạp hơn so với perceptron tuyến tính.

- Linh hoạt và đa dạng: MLPs có thể được xây dựng với nhiều lớp ẩn và kích thước lớp đầu ra linh hoạt. Điều này cho phép chúng có khả năng học và biểu diễn các mô hình phức tạp hơn, với khả năng tìm ra các đặc trưng phức tạp và tương quan giữa các đặc trưng.

- Học được tự động các đặc trưng: Với các lớp ẩn và hàm kích hoạt phi tuyến, MLPs có khả năng tự động học các đặc trưng từ dữ liệu đầu vào mà không cần phải xác định trước các đặc trưng cần tìm kiếm. Điều này giúp MLPs linh hoạt trong việc học và biểu diễn các mô hình phức tạp và không gian đặc trưng lớn.

- Khả năng học từ dữ liệu lớn: MLPs có khả năng học từ dữ liệu lớn và phụ thuộc vào thuật toán gradient descent để điều chỉnh các trọng số. Với sự phát triển của các phương pháp tối ưu hóa và tính toán hiệu suất cao, MLPs có thể được huấn luyện trên các tập dữ liệu lớn và phức tạp.

### Achitecture of MLP

Kiến trúc của MLP (Multilayer Perceptron) bao gồm nhiều lớp nút (neurons) kết nối với nhau, còn được gọi là các lớp. Thông thường, nó bao gồm ba loại lớp: lớp đầu vào, một hoặc nhiều lớp ẩn và lớp đầu ra. Mỗi lớp bao gồm một tập hợp các nút, và các nút trong các lớp liền kề được kết nối thông qua các kết nối có trọng số.

- Lớp đầu vào: Lớp đầu vào là lớp đầu tiên của MLP và đại diện cho dữ liệu đầu vào. Mỗi nút trong lớp đầu vào tương ứng với một đặc trưng hoặc thuộc tính của dữ liệu đầu vào. Số nút trong lớp đầu vào bằng với số chiều của dữ liệu đầu vào.

- Các lớp ẩn: Các lớp ẩn là các lớp trung gian giữa lớp đầu vào và lớp đầu ra. Chúng đóng vai trò quan trọng trong việc học các biểu diễn phức tạp và trích xuất các đặc trưng từ dữ liệu đầu vào. MLPs có thể có một hoặc nhiều lớp ẩn, và số nút trong mỗi lớp ẩn có thể thay đổi. Mỗi nút trong các lớp ẩn áp dụng hàm kích hoạt phi tuyến vào tổng có trọng số của đầu vào từ lớp trước đó.

- Lớp đầu ra: Lớp đầu ra là lớp cuối cùng của MLP và tạo ra đầu ra hoặc dự đoán dựa trên các đặc trưng đã học từ các lớp ẩn. Số nút trong lớp đầu ra phụ thuộc vào vấn đề cụ thể đang được giải quyết. Ví dụ, trong phân loại nhị phân, có thể có một nút duy nhất đại diện cho xác suất hoặc nhãn lớp. Trong phân loại đa lớp, sẽ có nhiều nút, mỗi nút tương ứng với một lớp khác nhau.

Các nút trong MLP được kết nối với nhau thông qua các kết nối có trọng số. Mỗi kết nối giữa các nút có một trọng số liên kết xác định sức mạnh của kết nối. Trong quá trình huấn luyện, trọng số các kết nối được điều chỉnh theo từng bước để giảm thiểu hàm mất mát, cho phép mạng học và đưa ra dự đoán chính xác.

### Activation function

Activation functions (hàm kích hoạt) là các hàm được áp dụng cho mỗi neuron trong mạng neural để giới hạn và điều chỉnh đầu ra của nút dựa trên tổng trọng số của đầu vào. Các hàm kích hoạt quyết định xem neuron có nên kích hoạt (được kích hoạt) hay không (không được kích hoạt) dựa trên giá trị đầu vào.

Có nhiều loại hàm kích hoạt phổ biến trong mạng neural, bao gồm:

- Hàm Sigmoid: Hàm sigmoid chuyển đổi giá trị đầu vào thành một giá trị trong khoảng (0, 1). Nó được sử dụng rộng rãi trong các lớp ẩn của MLP trước khi được thay thế bằng các hàm kích hoạt khác.

- Hàm Tanh: Hàm tanh chuyển đổi giá trị đầu vào thành một giá trị trong khoảng (-1, 1). Tương tự như sigmoid, hàm tanh cũng được sử dụng trong các lớp ẩn của MLP.

- Hàm ReLU (Rectified Linear Unit): Hàm ReLU giữ lại giá trị dương của đầu vào và đưa ra 0 cho các giá trị âm. Nó đã trở thành một lựa chọn phổ biến trong các mạng neural sâu vì tính đơn giản và khả năng giải quyết vấn đề gradient mất mát.

- Hàm Leaky ReLU: Hàm Leaky ReLU tương tự như ReLU, nhưng cho phép các giá trị âm nhỏ thông qua một hệ số rò rỉ nhỏ. Điều này giúp khắc phục vấn đề "neuron chết" có thể xảy ra với ReLU.

- Hàm Softmax: Hàm softmax được sử dụng trong lớp đầu ra của mạng neural đa lớp để tạo ra một phân phối xác suất trên các lớp đầu ra. Nó thường được sử dụng trong các bài toán phân loại đa lớp.

### Backpropagation

Backpropagation (phương pháp truyền ngược) là một thuật toán quan trọng trong huấn luyện mạng neural. Nó được sử dụng để tính toán và điều chỉnh các trọng số của mạng neural dựa trên độ lỗi (loss) giữa đầu ra dự đoán và đầu ra thực tế.

Thuật toán backpropagation hoạt động theo các bước sau:

- Feedforward: Dữ liệu được truyền qua mạng neural từ lớp đầu vào cho đến lớp đầu ra. Mỗi nút tính tổng trọng số của đầu vào từ lớp trước và áp dụng hàm kích hoạt để tính toán đầu ra.

- Tính toán lỗi: Đầu ra dự đoán được so sánh với đầu ra thực tế để tính toán độ lỗi. Công thức lỗi phụ thuộc vào loại vấn đề và mục tiêu huấn luyện.

- Backpropagation: Độ lỗi được truyền ngược từ lớp đầu ra về lớp đầu vào. Đối với mỗi nút, độ lỗi được phân phối lại theo trọng số của các kết nối đến nút đó. Điều này tính toán độ lỗi riêng cho mỗi trọng số.

- Cập nhật trọng số: Sử dụng độ lỗi tính toán được, thuật toán gradient descent được áp dụng để cập nhật các trọng số của mạng neural. Mục tiêu là điều chỉnh trọng số để giảm thiểu độ lỗi và cải thiện hiệu suất của mạng.

Quá trình này được lặp lại cho tất cả các mẫu huấn luyện trong tập dữ liệu đến khi đạt được điều kiện dừng, chẳng hạn như số lượng lần lặp tối đa hoặc độ lỗi đạt một ngưỡng chấp nhận được.

### Output with activation function

Một Multi-Layer Perceptron (MLP) với nhiều đầu ra là một kiến trúc mạng neural có khả năng tạo ra nhiều đầu ra cho một đầu vào cụ thể. Nó bao gồm nhiều lớp của các nút, bao gồm lớp đầu vào, một hoặc nhiều lớp ẩn và lớp đầu ra.

Trong MLP với nhiều đầu ra, mỗi nút đầu ra trong lớp đầu ra tương ứng với một đầu ra hoặc nhãn lớp cụ thể. Các kích hoạt của các nút đầu ra này đại diện cho các dự đoán của mạng cho mỗi đầu ra.

Trong quá trình truyền thuận của mạng, dữ liệu đầu vào được truyền qua các lớp, với mỗi kích hoạt của nút được tính dựa trên tổng có trọng số của các đầu vào và hàm kích hoạt được chọn. Trong lớp cuối cùng, mỗi nút đầu ra tính toán kích hoạt của nó bằng cách sử dụng các đầu vào nhận được từ lớp trước đó.

Trong quá trình huấn luyện, các trọng số của mạng được điều chỉnh bằng cách sử dụng các kỹ thuật như backpropagation và gradient descent để giảm sự khác biệt giữa các đầu ra dự đoán và đầu ra thực tế. Hàm mất mát được sử dụng cho quá trình huấn luyện có thể thay đổi tùy thuộc vào nhiệm vụ cụ thể, chẳng hạn như mean squared error cho các bài toán hồi quy hoặc cross-entropy loss cho các bài toán phân loại.

### Softmax

Hàm softmax (softmax function) là một hàm toán học phổ biến được sử dụng trong mạng neural để biến đổi đầu ra của một lớp thành một phân phối xác suất.

Hàm softmax thường được áp dụng cho lớp đầu ra của một mô hình phân loại đa lớp, trong đó mỗi lớp đại diện cho một nhãn khác nhau. Đầu vào của hàm softmax là một vector, và đầu ra là một vector có cùng kích thước, trong đó mỗi phần tử đại diện cho xác suất của một lớp cụ thể.

## Applications of Neural Networks

### Convolutional Neural Network(CNN)

CNN là viết tắt của Convolutional Neural Network (Mạng Neural tích chập), là một loại kiến trúc mạng neural sử dụng phổ biến trong lĩnh vực xử lý hình ảnh và thị giác máy tính. CNN được thiết kế để tự động học và trích xuất đặc trưng từ dữ liệu hình ảnh.

CNN được cấu trúc dựa trên sự kết hợp giữa các lớp tích chập (convolutional layers), lớp gộp (pooling layers) và các lớp kết nối đầy đủ (fully connected layers). Lớp tích chập được sử dụng để tìm ra các đặc trưng cục bộ trong hình ảnh bằng cách áp dụng các bộ lọc thông qua việc tích chập trên toàn bộ hình ảnh hoặc các vùng cục bộ. Lớp gộp được sử dụng để giảm kích thước của đặc trưng và tạo ra một biểu diễn đặc trưng tổng quát hơn. Các lớp kết nối đầy đủ được sử dụng để phân loại các đặc trưng đã được trích xuất.

Điểm mạnh của CNN là khả năng tự động học các đặc trưng cấp cao từ dữ liệu hình ảnh, mà không cần chỉ định các đặc trưng cụ thể trước. Nó có khả năng nhận diện và hiểu các đặc trưng phức tạp trong hình ảnh, như các biên, góc, vùng tối sáng, và các hình dạng phức tạp hơn.

CNN đã đạt được nhiều thành công đáng kể trong nhiều nhiệm vụ xử lý hình ảnh, bao gồm nhận dạng đối tượng, phân loại hình ảnh, nhận dạng khuôn mặt, và nhiều ứng dụng khác. Nó đã trở thành một trong những công cụ quan trọng và mạnh mẽ trong lĩnh vực thị giác máy tính và AI.

### Convolutional layer

Convolutional layer (lớp tích chập) là một trong những thành phần chính của Convolutional Neural Network (CNN). Lớp tích chập được sử dụng để trích xuất đặc trưng từ dữ liệu đầu vào, như hình ảnh, bằng cách áp dụng phép tích chập.

Lớp tích chập bao gồm một tập hợp các bộ lọc (filter) hoặc kernel, mỗi bộ lọc có kích thước nhỏ và được áp dụng lên các phần nhỏ của dữ liệu đầu vào. Bộ lọc này di chuyển theo từng bước (stride) trên toàn bộ dữ liệu đầu vào và thực hiện phép tích chập để tạo ra các đặc trưng cục bộ.

Trong quá trình tích chập, bộ lọc sẽ nhân từng phần tử của nó với các phần tương ứng trong vùng chồng lên của dữ liệu đầu vào và tính tổng các kết quả nhân để tạo ra một giá trị đầu ra duy nhất. Quá trình này được thực hiện cho từng vị trí của bộ lọc trên toàn bộ dữ liệu đầu vào, tạo ra một ma trận đầu ra gọi là feature map (bản đồ đặc trưng).

Lớp tích chập có thể có nhiều bộ lọc, mỗi bộ lọc sẽ tạo ra một feature map riêng. Các feature map này biểu thị các đặc trưng cụ thể được trích xuất từ dữ liệu đầu vào. Các bộ lọc trong lớp tích chập có thể học được các đặc trưng như biên, góc, hoặc các hình dạng phức tạp hơn thông qua quá trình huấn luyện mạng.

Lớp tích chập thường kết hợp với các phép kích hoạt phi tuyến (non-linear activation) như ReLU (Rectified Linear Unit) để tạo ra đầu ra phi tuyến tính và tăng khả năng học các đặc trưng phức tạp.

Các lớp tích chập liên tiếp nhau trong CNN tạo thành một kiến trúc đa tầng, cho phép mạng học các đặc trưng cấp cao từ dữ liệu đầu vào. Quá trình này góp phần trong việc nhận diện, phân loại, hay xử lý thông tin trong các tác vụ thị giác máy tính.

### How to compute output size?

Để tính kích thước đầu ra của một lớp tích chập trong CNN, ta sử dụng các công thức sau:

Kích thước đầu ra (W') của lớp tích chập trong chiều rộng (width):
W' = (W - F + 2P) / S + 1

Trong đó:

W là kích thước đầu vào trong chiều rộng.
F là kích thước của bộ lọc (filter) trong chiều rộng.
P là padding (đệm), là số lượng hàng/cột thêm vào xung quanh đầu vào. Nếu không sử dụng padding, P = 0.
S là bước di chuyển (stride), là số lượng hàng/cột bước nhảy khi áp dụng bộ lọc. Nếu không sử dụng stride, S = 1.
Kích thước đầu ra (H') của lớp tích chập trong chiều cao (height):
H' = (H - F + 2P) / S + 1

Trong đó:

H là kích thước đầu vào trong chiều cao.
Số kênh đầu ra (C') của lớp tích chập:
C' là số lượng bộ lọc được áp dụng trong lớp tích chập.

### Pooling layer
Pooling layer (lớp gộp) là một thành phần quan trọng trong Convolutional Neural Network (CNN). Lớp gộp được sử dụng để giảm kích thước không gian của đầu vào, giữ lại thông tin quan trọng và giảm độ phức tạp tính toán.

Lớp gộp hoạt động bằng cách áp dụng một phép gộp lên các vùng không gian của dữ liệu đầu vào. Phép gộp thường là phép lấy giá trị lớn nhất (max pooling) hoặc phép lấy giá trị trung bình (average pooling) của các giá trị trong vùng.

Khi áp dụng max pooling, lớp gộp chia dữ liệu đầu vào thành các vùng không chồng lên nhau và chọn giá trị lớn nhất trong mỗi vùng làm giá trị đại diện. Điều này giúp giữ lại thông tin quan trọng và đặc trưng nổi bật trong dữ liệu.

Khi áp dụng average pooling, lớp gộp tính giá trị trung bình của các giá trị trong mỗi vùng và sử dụng giá trị này làm giá trị đại diện cho vùng. Điều này giúp giảm độ phức tạp tính toán và làm mờ các đặc trưng cục bộ.

Lớp gộp thường được áp dụng sau các lớp tích chập trong CNN. Việc sử dụng lớp gộp giúp giảm kích thước không gian của đầu vào, làm giảm số lượng tham số và tính toán, và tạo ra một biểu diễn tổng quát hơn cho các đặc trưng. Điều này giúp cải thiện tính tổng quát và ổn định của mô hình.

### Type of pooling layer
Có hai loại phổ biến của lớp gộp (pooling layer) trong Convolutional Neural Network (CNN):

- Max Pooling: Max pooling là phép gộp thông qua việc chọn giá trị lớn nhất từ một vùng không gian nhất định trong feature map. Phép gộp này giữ lại thông tin quan trọng nhất và giúp giảm kích thước của feature map. Vùng gộp thường có kích thước và bước đi (stride) được chỉ định trước. Max pooling thường được sử dụng để tìm ra các đặc trưng nổi bật trong ảnh.

- Average Pooling: Average pooling là phép gộp thông qua việc tính trung bình của các giá trị trong một vùng không gian nhất định trong feature map. Phép gộp này tính trung bình các giá trị và giúp giảm độ phức tạp của mạng. Average pooling thường được sử dụng để làm mờ các đặc trưng cục bộ và tạo ra biểu diễn tổng quát hơn cho ảnh.

### Summary on CNN
Convolutional Neural Network (CNN) là một mạng nơ-ron nhân tạo được sử dụng phổ biến trong xử lý ảnh và nhận dạng đối tượng. CNN được thiết kế dựa trên cấu trúc và cách hoạt động của thị giác con người.

Các thành phần chính trong CNN bao gồm lớp tích chập (convolutional layer), lớp gộp (pooling layer), và lớp kết nối đầy đủ (fully connected layer).

Lớp tích chập áp dụng các bộ lọc (filter) trên đầu vào để tạo ra các đặc trưng cục bộ. Bằng cách trượt bộ lọc trên toàn bộ đầu vào, các kết quả tích chập được tạo ra, gọi là feature map, chứa thông tin về các đặc trưng cục bộ của hình ảnh.

Lớp gộp được sử dụng để giảm kích thước không gian của feature map. Phép gộp (max pooling hoặc average pooling) được áp dụng để lấy thông tin quan trọng từ các vùng không gian và tạo ra các feature map có kích thước nhỏ hơn.

Sau các lớp tích chập và gộp, thông tin được truyền vào lớp kết nối đầy đủ để thực hiện việc phân loại. Lớp kết nối đầy đủ gồm các nơ-ron liên kết tất cả các đặc trưng để dự đoán đầu ra.

Quá trình huấn luyện của CNN bao gồm quá trình lan truyền thuận (forward propagation) để tính toán đầu ra dựa trên các tham số mạng, và quá trình lan truyền ngược (backpropagation) để điều chỉnh các tham số mạng dựa trên độ lỗi.

CNN đã đạt được kết quả ấn tượng trong nhiều bài toán như nhận dạng ảnh, phân loại đối tượng, nhận dạng khuôn mặt và nhận dạng chữ viết tay. Với cấu trúc và khả năng học tự động của nó, CNN là một công cụ mạnh mẽ trong xử lý ảnh và trí tuệ nhân tạo.

### Popular neural networks
Popular network architectures
- LeNet
- AlexNet
- VGG16
- Inception v1, v3
- ResNet-50
- MobileNe
