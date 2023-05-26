# Clustering

Phân cụm (Clustering) là một phương pháp trong machine learning (học máy) được sử dụng để nhóm các điểm dữ liệu có tính chất tương đồng lại với nhau. Mục tiêu của phân cụm là tìm ra cấu trúc ẩn trong dữ liệu mà không cần có nhãn hay thông tin trước.

Trong phân cụm, chúng ta xem xét các điểm dữ liệu và cố gắng tìm ra cách chia chúng thành các nhóm (clusters) dựa trên mức độ tương đồng giữa chúng. Mỗi nhóm sẽ chứa các điểm dữ liệu có sự tương đồng cao với nhau, trong khi giữa các nhóm thì có sự khác biệt.

Có nhiều phương pháp phân cụm khác nhau, nhưng hai phương pháp phân cụm phổ biến là K-Means và Hierarchical Clustering. Trong K-Means, chúng ta chọn một số lượng K centroid ban đầu, sau đó lặp lại quá trình gán điểm vào cluster và cập nhật centroid cho đến khi sự hội tụ. Hierarchical Clustering xây dựng các cấu trúc phân cấp dựa trên sự tương đồng giữa các điểm dữ liệu, bắt đầu từ việc mỗi điểm là một cluster riêng và sau đó kết hợp các cluster lại với nhau dựa trên mức độ tương đồng.

Các phương pháp phân cụm có thể được đánh giá bằng cách sử dụng các độ đo như Silhouette coefficient hoặc Index Dunn để đo lường mức độ phân cụm tốt.

Phân cụm là một công cụ hữu ích trong machine learning và được áp dụng trong nhiều lĩnh vực như phân loại hình ảnh, phân loại văn bản, phân tích khách hàng và nhiều ứng dụng khác. Nó giúp chúng ta hiểu rõ hơn về cấu trúc dữ liệu và khám phá thông tin ẩn chưa được biết đến trước đó.

## What Is Cluster Analysis?

Phân tích cụm, còn được gọi là phân cụm, là một kỹ thuật phân tích dữ liệu được sử dụng để xác định các nhóm hoặc cụm trong một tập dữ liệu. Đây là một phương pháp học không giám sát nhằm khám phá các mẫu, cấu trúc hoặc mối quan hệ tự nhiên trong dữ liệu dựa trên sự tương đồng hoặc gần gũi giữa các quan sát.

Mục tiêu của phân tích cụm là nhóm các điểm dữ liệu tương tự lại với nhau trong khi giữ cho các điểm khác biệt riêng rẽ. Sự tương tự hoặc khác biệt giữa các điểm dữ liệu thường được đo bằng các độ đo khoảng cách như khoảng cách Euclid, khoảng cách Manhattan hoặc độ tương tự cosin.

Phân tích cụm có thể được áp dụng trong nhiều lĩnh vực và ứng dụng, bao gồm:

- Phân đoạn thị trường: Xác định các nhóm khách hàng có các đặc điểm và hành vi tương tự nhau để điều chỉnh chiến lược tiếp thị.

- Nhận dạng hình ảnh và đối tượng: Nhóm hình ảnh hoặc đối tượng tương tự dựa trên các đặc trưng hình ảnh.

- Phát hiện dấu hiệu bất thường: Xác định các mẫu không bình thường hoặc các giá trị ngoại lệ khác biệt so với quy tắc thông thường.

- Phân cụm tài liệu: Tổ chức các tài liệu thành các chủ đề hoặc đề tài có ý nghĩa.

- Phân đoạn khách hàng: Nhóm khách hàng dựa trên các thói quen mua hàng và sở thích để cải thiện công tác tiếp thị cá nhân.

- Phân cụm gen: Nhóm các gen hoặc chuỗi DNA dựa trên sự tương tự để hiểu các mẫu di truyền.

Có một số thuật toán được sử dụng trong phân tích cụm, bao gồm phân cụm K-means, phân cụm phân cấp, phân cụm dựa trên mật độ (DBSCAN) và mô hình hỗn hợp Gaussian (GMM). Mỗi thuật toán có phương pháp riêng để xác định các cụm và gán các điểm dữ liệu vào chúng.

Khoảng cách Euclid: Khoảng cách Euclid là độ đo khoảng cách phổ biến nhất. Nó tính toán khoảng cách theo đường thẳng giữa hai điểm trong không gian Euclid. Đối với hai điểm (x1, y1) và (x2, y2) trong không gian hai chiều, khoảng cách Euclid được tính bằng công thức:

d = căn bậc hai của ((x2 - x1)^2 + (y2 - y1)^2)

Độ đo này thích hợp cho các biến liên tục và giả định rằng các biến có mối quan hệ tuyến tính.

Khoảng cách Manhattan: Khoảng cách Manhattan, còn được gọi là khoảng cách thành phố hoặc khoảng cách L1, tính tổng các khác biệt tuyệt đối giữa các tọa độ tương ứng của hai điểm. Nó được gọi là khoảng cách Manhattan vì nó đo khoảng cách như khi đi dọc theo các khối thành phố. Đối với hai điểm (x1, y1) và (x2, y2), khoảng cách Manhattan được tính bằng công thức:

d = |x2 - x1| + |y2 - y1|

Khoảng cách Manhattan hữu ích cho các biến rời rạc hoặc khi mối quan hệ giữa các biến là phi tuyến.

## Partitioning Methods

### K-means Clustering

Phân cụm K-means là một thuật toán phân cụm phổ biến trong lĩnh vực học không giám sát. Nó được sử dụng để phân nhóm các điểm dữ liệu trong một tập dữ liệu thành các cụm dựa trên sự tương đồng giữa chúng.

Cách hoạt động của thuật toán K-means như sau:

- Chọn số lượng cụm K mà bạn muốn tạo ra.
- Chọn ngẫu nhiên K điểm trong tập dữ liệu ban đầu làm điểm trung tâm ban đầu cho các cụm.
- Lặp lại các bước sau cho đến khi điều kiện dừng được đáp ứng:

  - Gán mỗi điểm dữ liệu vào cụm có trung tâm gần nhất bằng cách tính khoảng cách từ điểm dữ liệu đến các trung tâm cụm và chọn trung tâm gần nhất.
  - Cập nhật trung tâm của mỗi cụm bằng cách tính trung bình của các điểm dữ liệu trong cụm.

- Kết quả là tập hợp các cụm, với mỗi điểm dữ liệu được gán vào một cụm cụ thể.

Thuật toán K-means cố gắng tìm ra các trung tâm cụm tối ưu nhằm tối thiểu hóa tổng bình phương khoảng cách giữa các điểm dữ liệu và trung tâm của cụm tương ứng. Phân cụm K-means rất hiệu quả khi áp dụng cho các tập dữ liệu lớn và có thể mở rộng để xử lý các biến liên tục và rời rạc.

Tuy nhiên, K-means cần chỉ định trước số lượng cụm K và dễ bị ảnh hưởng bởi các điểm dữ liệu ngoại lệ. Đồng thời, nó chỉ tìm ra các cụm có hình dạng hình cầu và không phù hợp cho các cấu trúc phân phối dữ liệu phức tạp.

### Problem with Selecting Initial Points

Vấn đề với việc chọn điểm khởi tạo ban đầu trong thuật toán K-means được gọi là "problems with selecting initial points". Khi thực hiện phân cụm K-means, việc chọn điểm ban đầu có thể ảnh hưởng đến kết quả cuối cùng của thuật toán.

Có một số vấn đề có thể xảy ra khi chọn điểm khởi tạo ban đầu:

- Phụ thuộc vào điểm khởi tạo ban đầu: K-means có thể cố gắng tìm ra một điểm dừng cục bộ (local optimum) thay vì tìm ra giải pháp tối ưu toàn cục (global optimum). Điểm khởi tạo ban đầu có thể ảnh hưởng đến điểm dừng cục bộ mà thuật toán tìm thấy. Kết quả cuối cùng có thể khác nhau tùy thuộc vào cách chọn điểm ban đầu.

- Ảnh hưởng của nhiễu và điểm dữ liệu ngoại lệ: Nếu điểm khởi tạo ban đầu được chọn gần các điểm dữ liệu nhiễu hoặc điểm dữ liệu ngoại lệ, nó có thể ảnh hưởng đến quá trình hội tụ của thuật toán. Các điểm dữ liệu nhiễu có thể làm sai lệch vị trí của các trung tâm cụm.

- Điểm khởi tạo không đại diện cho dữ liệu: Nếu điểm khởi tạo ban đầu được chọn không đại diện cho cấu trúc dữ liệu thực tế, nó có thể dẫn đến phân cụm không chính xác hoặc không thể tìm ra cấu trúc chính xác của dữ liệu.

Để giải quyết vấn đề này, một phương pháp là chạy thuật toán K-means nhiều lần với các điểm khởi tạo ngẫu nhiên khác nhau và chọn kết quả tốt nhất. Một cách khác là sử dụng các phương pháp khởi tạo điểm ban đầu nâng cao như K-means++ để chọn các điểm khởi tạo có phân bố tốt hơn trong không gian dữ liệu.

### Solutions to Inital Centroids Problem

Có một số giải pháp để khắc phục vấn đề với việc chọn điểm khởi tạo ban đầu trong thuật toán K-means:

- K-means++: K-means++ là một phương pháp khởi tạo điểm ban đầu cho thuật toán K-means. Thay vì chọn các điểm khởi tạo ban đầu hoàn toàn ngẫu nhiên, K-means++ thực hiện một quá trình lựa chọn thông minh để chọn các điểm khởi tạo ban đầu có phân bố tốt hơn trong không gian dữ liệu. Phương pháp này giúp đạt được kết quả tốt hơn và tăng khả năng thuật toán hội tụ nhanh hơn.

- Chạy nhiều lần với các điểm khởi tạo ngẫu nhiên: Một phương pháp khác để khắc phục vấn đề với điểm khởi tạo ban đầu là chạy thuật toán K-means nhiều lần với các điểm khởi tạo ngẫu nhiên khác nhau. Sau đó, chọn kết quả tốt nhất dựa trên một tiêu chí như giá trị hàm mục tiêu (sum of squared distances) thấp nhất hoặc độ tương tự giữa các phân cụm.

- Phương pháp chọn trung tâm từ dữ liệu đại diện: Một phương pháp khác là chọn trung tâm ban đầu từ các điểm dữ liệu đại diện trong tập dữ liệu. Có thể sử dụng các phương pháp phân cụm trước đó như phân cụm hiển thị (hierarchical clustering) hoặc phân cụm GMM (Gaussian Mixture Model) để chọn một số điểm dữ liệu đại diện và sử dụng chúng làm điểm khởi tạo ban đầu.

- Chọn trung tâm ban đầu theo kiến thức chuyên gia: Trong một số trường hợp, kiến thức chuyên gia về dữ liệu có thể được sử dụng để chọn các điểm khởi tạo ban đầu. Kiến thức này có thể bao gồm các thông tin về cấu trúc dữ liệu, nhãn, hoặc đặc trưng quan trọng. Bằng cách sử dụng kiến thức này, có thể chọn các điểm khởi tạo ban đầu gần với các trung tâm dự kiến của các cụm hoặc đại diện cho các đặc trưng quan trọng của dữ liệu.
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/075f21c6-3975-4d57-9b44-cad0f1d188e0)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/82be6fe4-e569-4006-9f9a-6e7ec05dedcd)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/c6e1cc02-b29a-4649-86f9-fe281955414e)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/b3d38c56-c813-4888-b0d1-c060381341c6)
![image](https://github.com/trinhvu1711/ML_Semester2_2023/assets/81180330/73a2cee8-e8bb-45f3-840d-9d17f16574d2)

## Hierarchical Methods

## Density-and Grid-Based Methods

## Evaluation of Clustering

## Summary
