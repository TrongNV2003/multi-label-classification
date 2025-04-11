import json

# Danh sách nhãn của bạn
label_list = [
    'Đồng ý', 'Chào hỏi', 'Cung cấp vị trí, địa chỉ giao hàng', 'Hỏi vị trí khách nhận hàng', 
    'Cung cấp tên sản phẩm', 'Hỏi tình trạng nhận hàng', 'Cung cấp mã đơn hàng', 'Thông báo đã đến nơi', 
    'Cung cấp thời gian giao hàng', 'Cung cấp giá sản phẩm', 'Chuyển khoản', 'Cung cấp chi phí đơn hàng', 
    'Mô tả đường đi', 'Hỏi tên sản phẩm', 'Nhận hàng được', 'Gửi tại nhà và nơi làm việc', 
    'Yêu cầu nhắc lại thông tin', 'Cung cấp tên khách hàng', 'Cảm thán', 'Do không có nhà', 
    'Cung cấp vị trí shipper', 'Cung cấp vị trí khách đang đứng', 'Không đồng ý', 'Hỏi thời gian giao hàng', 
    'Đóng góp', 'Giao cho người thân', 'Hỏi người nhận hàng', 'Hỏi chi phí đơn hàng', 'Hỏi vị trí shipper', 
    'Cung cấp vị trí Tài xế', 'Gọi trước khi giao', 'Không nhận hàng được', 'Cung cấp số lượng sản phẩm', 
    'Do khách hàng huỷ đơn', 'Cho xem hàng', 'Do đi làm', 'Hỏi vị trí đặt hàng', 'Cung cấp tên shop', 
    'Không hài lòng', 'Đặt trước cửa', 'Do gọi khách nhiều lần nhưng không bắt máy', 'Hỏi vị trí tài xế', 
    'Giao cho hàng xóm', 'Giao cho bảo vệ', 'Do khách hàng từ chối nhận', 'Cung cấp tên sàn thương mại điện tử', 
    'Do khách hàng không có nhà', 'Cung cấp tên đơn vị vận chuyển', 'Do hàng khác với mô tả', 
    'Hàng khác với mô tả', 'Cung cấp số điện thoại khách hàng', 'Cho thử hàng', 'Hỏi tên shop', 
    'Hỏi số lượng sản phẩm', 'Giao cho chính khách hàng', 'Hỗ trợ đổi trả hàng tại chỗ', 
    'Cung cấp vị trí tài xế', 'Hỏi vị trí Tài xế', 'Cung cấp kích thước sản phẩm', 'Cung cấp chính sách nhận hàng', 
    'Thanh toán trả sau', 'Hỏi hình thức thanh toán', 'Đồng kiểm ngoại quan', 'Hỗ trợ hình ảnh, video', 
    'Bằng tiền mặt', 'Hỗ trợ bê vác', 'Nhận hàng một phần', 'Nhắn tin trước khi giao', 'Giục giao hàng', 
    'Gửi vào kho', 'Đồng kiểm chi tiết', 'Nhầm hàng', 'Hài lòng', 'Hướng dẫn gửi hàng', 
    'Hỏi tên đơn vị vận chuyển', 'Hướng dẫn quy trình đổi trả sản phẩm', 'Đổi size', 'Thanh toán trả trước', 
    'Do về quê', 'Hướng dẫn thao tác app', 'Hỏi kích thước sản phẩm', 'Do shop huỷ đơn hàng', 
    'Do trùng đơn hàng', 'Do sai đơn hàng', 'Hỏi quy trình đổi trả sản phẩm', 'Hàng hỏng vỡ', 
    'Yêu cầu hoàn tiền', 'Hỏi tên sàn thương mại điện tử', 'Thiếu hàng', 'Hỏi nguồn gốc sản phẩm', 
    'Bấm chuông', 'Cung cấp cân nặng sản phẩm', 'Cung cấp chương trình giảm giá, khuyến mại', 
    'Do giao sai đơn', 'Do khách hàng nhận hàng tại bưu cục', 'Cung cấp chính sách hoàn tiền', 
    'Đổi mẫu mã', 'Hỏi cân nặng sản phẩm', 'Giao cho lễ tân', 'Đổi màu', 'Do hàng lỗi', 
    'Cung cấp thương hiệu sản phẩm', 'Sai cân', 'Hỏi chính sách nhận hàng', 'Hỏi SĐT shop', 
    'Nhận hàng toàn bộ', 'Quét mã QR', 'Đồng kiểm mã imei', 'Hỏi chương trình giảm giá, khuyến mại', 
    'Do gặp sự cố', 'Hướng dẫn chia sẻ định vị', 'Hỏi tên thương hiệu sản phẩm', 'Do lạc đường', 
    'Hướng dẫn quy trình huỷ đơn hàng', 'Hỏi quy trình huỷ đơn hàng', 'Hướng dẫn thanh toán', 
    'Cung cấp chính sách bảo hành sản phẩm', 'Do hàng hỏng vỡ', 'Hỏi số điện thoại shop', 
    'Gửi tủ locker', 'Giục trả hàng', 'Đồng kiểm chứng từ', 'Hỏi chính sách hoàn tiền', 
    'Gửi vào hộp thư', 'Không bấm chuông', 'Thanh toán qua ví', 'Hỏi chính sách bảo hành sản phẩm', 
    'Hỗ trợ lắp đặt', 'Gõ cửa', 'Hàng thất lạc', 'Thanh toán bằng thẻ tín dụng', 'Do tắc đường', 
    'Đồng kiểm date', 'Ướt hàng', 'Hỏi đánh giá chất lượng sản phẩm', 'Do đi du lịch'
]

# Chuyển thành danh sách các dictionary
json_data = [{"intent": label} for label in label_list]

# Cách 1: In ra dưới dạng chuỗi như bạn yêu cầu
output_lines = [f"'    {{\"intent\": \"{label}\"}},\\n'" for label in label_list]
output_str = "[\n" + "\n".join(output_lines) + "\n]"

# In ra màn hình
print(output_str)

# (Tùy chọn) Lưu vào file nếu cần
output_file = "intent_list_exact_syntax.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_str)
print(f"Đã lưu vào file: {output_file}")