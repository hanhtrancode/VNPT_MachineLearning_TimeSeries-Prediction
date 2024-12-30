# VNPT_MachineLearning_TimeSeries_Prediction

Trong bối cảnh chuyển đổi số và nhu cầu nâng cao hiệu quả dịch vụ công trực tuyến, việc dự đoán chính xác thời gian hoàn thành các thủ tục hành chính đóng vai trò quan trọng. Điều này không chỉ giúp cải thiện trải nghiệm người dùng mà còn tối ưu hóa quy trình vận hành của các đơn vị thực hiện.

Dựa trên dữ liệu thu thập được từ nhiều nhóm đơn vị và thủ tục khác nhau, tôi đã triển khai một dự án sử dụng mô hình XGBoost kết hợp với chiến lược dự báo đệ quy (recursive prediction) để giải quyết bài toán multi-time series forecasting. Quy trình này được thiết kế nhằm xử lý dữ liệu không đồng nhất và không ổn định về tần suất, đảm bảo mô hình có thể dự đoán chính xác và áp dụng linh hoạt trên nhiều nhóm khác nhau.

Từ việc tiền xử lý dữ liệu phức tạp, chia tập train/test thông minh, đến tối ưu hóa hyperparameter và đánh giá kết quả, tôi đã xây dựng một pipeline mạnh mẽ để dự đoán thời gian hoàn thành thủ tục với độ chính xác cao. Hãy cùng khám phá chi tiết quy trình và những kết quả nổi bật mà tôi đã đạt được!

<!DOCTYPE html>
<html>
<head>
    <title>Time-Series Prediction Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2 {
            color: #333;
        }
        h1 {
            text-align: center;
        }
        ul {
            margin: 0;
            padding: 0 20px;
        }
        li {
            margin-bottom: 10px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
        }
        .highlight {
            color: #0066cc;
        }
    </style>
</head>
<body>
    <h1>Time-Series Prediction Project</h1>
    <h2>1. Khám phá và Xử lý Dữ liệu (Exploratory Data Analysis - EDA)</h2>
    <p><b>Dữ liệu ban đầu:</b></p>
    <ul>
        <li>Gồm các cột: <code>ngay_gio_tiep_nhan</code>, <code>ma_don_vi_thuc_hien</code>, <code>ma_thu_tuc</code>, <code>Thoi_gian_xu_ly</code> (mục tiêu cần dự đoán).</li>
        <li>Nhiều nhóm khác nhau (<code>ma_don_vi_thuc_hien</code> và <code>ma_thu_tuc</code>), mỗi nhóm là một chuỗi thời gian độc lập.</li>
    </ul>
    <p><b>Thách thức trong dữ liệu:</b></p>
    <ul>
        <li>Tần suất thời gian không cố định, không thể sử dụng các mô hình dự báo truyền thống như ARIMA hay Prophet.</li>
        <li>Nhiều giá trị thiếu (missing values) và trùng lặp trong dữ liệu.</li>
    </ul>
    <p><b>Xử lý dữ liệu:</b></p>
    <ul>
        <li>Xử lý giá trị thiếu bằng phương pháp nội suy hoặc loại bỏ.</li>
        <li>Tạo thêm các đặc trưng thời gian (ngày trong tuần, giờ trong ngày, khoảng cách thời gian giữa các sự kiện trước đó).</li>
        <li>Tạo <i>lag features</i> để đưa thông tin lịch sử vào mô hình, ví dụ: thời gian xử lý của các thủ tục gần nhất.</li>
    </ul>
    <h2>2. Chiến lược Chia Dữ liệu</h2>
    <p><b>Tách tập Train/Validation/Test:</b></p>
    <ul>
        <li><b>Train-Validation:</b> Tập dữ liệu từ 01/01/2023 đến 01/11/2023.
            <ul>
                <li>Recursive Split: 3 tháng đầu tiên làm train, 1 tháng kế tiếp làm validation. Quy trình này lặp lại đến hết 01/11/2023.</li>
            </ul>
        </li>
        <li><b>Test:</b> Dữ liệu từ 01/11/2023 đến 31/12/2023.</li>
    </ul>
    <p><b>Đảm bảo không bị rò rỉ dữ liệu (data leakage):</b></p>
    <ul>
        <li>Giữ nguyên thứ tự thời gian giữa các tập.</li>
        <li>Không sử dụng thông tin của thủ tục trong tương lai để dự đoán kết quả của hiện tại.</li>
    </ul>
    <h2>3. Triển khai Mô hình XGBoost</h2>
    <p><b>XGBoost cho Time-Series:</b></p>
    <ul>
        <li><b>Đặc trưng đầu vào:</b>
            <ul>
                <li>Các đặc trưng thời gian: tháng, tuần, giờ, ngày nghỉ lễ.</li>
                <li>Các đặc trưng nhóm: <code>ma_don_vi_thuc_hien</code>, <code>ma_thu_tuc</code> (mã hóa bằng One-Hot hoặc Label Encoding).</li>
                <li><i>Lag features</i> và thống kê cuộn (rolling statistics): thời gian xử lý trung bình, max/min của các thủ tục gần đây.</li>
            </ul>
        </li>
        <li><b>Chiến lược dự báo:</b> Recursive Forecasting: Dự đoán từng bước thời gian (step-by-step), trong đó dự đoán của bước trước được sử dụng làm input cho bước tiếp theo.</li>
    </ul>
    <p><b>Tìm kiếm Hyperparameter:</b></p>
    <ul>
        <li>Sử dụng <code>RandomizedSearchCV</code> trên các siêu tham số của XGBoost như <code>max_depth</code>, <code>eta</code>, <code>gamma</code>, <code>subsample</code>, <code>colsample_bytree</code>.</li>
        <li>Sử dụng tập validation để chọn ra bộ tham số tốt nhất, tối ưu hóa theo MAE (Mean Absolute Error).</li>
    </ul>
    <h2>4. Đánh Giá và Hiệu Chỉnh</h2>
    <p><b>Đánh giá mô hình:</b></p>
    <ul>
        <li>Sử dụng MAE (Mean Absolute Error) và RMSE (Root Mean Squared Error) để đo độ chính xác.</li>
        <li>So sánh với baseline: dự đoán trung bình và dự đoán giá trị gần nhất.</li>
        <li>Phân tích kết quả theo từng nhóm <code>ma_don_vi_thuc_hien</code> và <code>ma_thu_tuc</code>.</li>
    </ul>
    <p><b>Xử lý Overfitting:</b></p>
    <ul>
        <li>So sánh MAE trên tập train và validation.</li>
        <li>Giảm độ phức tạp của mô hình qua việc điều chỉnh <code>max_depth</code>, <code>min_child_weight</code>, và <code>subsample</code>.</li>
    </ul>
    <h2>5. Triển khai Dự Báo</h2>
    <p><b>Kết quả Dự báo:</b></p>
    <ul>
        <li>Đồ thị so sánh giá trị thực tế và dự đoán cho từng nhóm <code>ma_don_vi_thuc_hien</code>.</li>
        <li>Tổng hợp lỗi dự đoán theo từng tháng.</li>
    </ul>
    <p><b>Triển khai mô hình:</b></p>
    <ul>
        <li>Sử dụng toàn bộ dữ liệu từ 01/01/2023 - 01/11/2023 để huấn luyện lại mô hình.</li>
        <li>Dự đoán tập test (01/11/2023 - 31/12/2023) và trình bày báo cáo cho khách hàng.</li>
    </ul>
    <h2>6. Bài Học Rút Ra</h2>
    <p><b>Ưu điểm:</b></p>
    <ul>
        <li>XGBoost hoạt động tốt trên các bài toán có nhiều đặc trưng đầu vào và không yêu cầu tần suất thời gian cố định.</li>
        <li>Chiến lược recursive forecasting giúp giải quyết bài toán multi-step với độ chính xác cao.</li>
    </ul>
    <p><b>Khó khăn:</b></p>
    <ul>
        <li>Xử lý dữ liệu không đồng nhất giữa các nhóm yêu cầu nhiều thời gian tiền xử lý.</li>
        <li>Recursive forecasting nhạy cảm với sai số tích lũy, cần tối ưu hóa kỹ lưỡng.</li>
    </ul>
    <p><b>Hướng phát triển:</b></p>
    <ul>
        <li>Tích hợp thêm các mô hình phi truyền thống như LightGBM hoặc LSTM để so sánh hiệu năng.</li>
        <li>Tự động hóa quy trình pipeline từ xử lý dữ liệu đến dự báo.</li>
    </ul>
</body>
</html>
