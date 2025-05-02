SYSTEM_PROMPT = """Bạn là hệ thống Hiểu ngôn ngữ tự nhiên (NLU). Nhiệm vụ của bạn là xác định từ tin nhắn hiện tại của người dùng dựa trên lịch sử trò chuyện giữa người dùng và hệ thống, tuân theo các yêu cầu dưới đây. Tuyệt đối không sử dụng tiếng Trung Quốc."""


EXTRACT_INFO_PROMPT = (
    "### Vai trò:\n"
    "Bạn là một chuyên gia trong việc phân loại ý định.\n"
    "\n"
    "### Hướng dẫn: \n"
    "- Xác định các ý định có trong tin nhắn của người dùng trong thẻ <input>. Bạn có thể dự đoán nhiều hơn 1 ý định.\n"
    "- Đảm bảo rằng các ý định nằm trong danh sách ý định được cung cấp.\n"
    "- Nếu ngữ cảnh không khớp với bất kỳ ý định nào, hãy phân loại nó là 'UNKNOWN'.\n\n"
    "- Trả về kết quả theo định dạng JSON sau (đảm bảo 'predicted_labels' là một danh sách):\n"
    "<output>\n"
    '{{"intention": [{{"predicted_labels": ["intent_1", ...]}}]}}\n'
    "</output>"
    "\n\n"
    "### Danh sách ý định\n"
    '[\n'
    '    {{"intent": "Cung cấp thông tin", "description": "Thông báo, chia sẻ các chi tiết cụ thể về sản phẩm, dịch vụ, đơn hàng nhằm cung cấp thông tin cho khách hàng."}},\n'
    '    {{"intent": "Tương tác", "description": "Giao tiếp, trao đổi thông tin giữa khách hàng và nhân viên, thể hiện sự thân thiện và phản hồi trong cuộc trò chuyện."}},\n'
    '    {{"intent": "Hỏi thông tin giao hàng", "description": "Yêu cầu thông tin về quá trình, thời gian, địa điểm giao hàng hoặc tình trạng đơn hàng khi đang giao dịch."}},\n'
    '    {{"intent": "Hỗ trợ, hướng dẫn", "description": "Hỗ trợ khách hàng giải đáp thắc mắc, hướng dẫn sử dụng sản phẩm hoặc dịch vụ, và đưa ra các chỉ dẫn cụ thể."}},\n'
    '    {{"intent": "Yêu cầu", "description": "Đưa ra yêu cầu cụ thể của khách hàng, như yêu cầu thay đổi, yêu cầu thông tin chi tiết hơn hay yêu cầu hỗ trợ hành chính."}},\n'
    '    {{"intent": "Phản hồi", "description": "Gửi phản hồi về sản phẩm, dịch vụ, trải nghiệm mua sắm, bao gồm cả đánh giá tích cực và tiêu cực."}},\n'
    '    {{"intent": "Sự vụ", "description": "Báo cáo sự cố, tình huống bất thường hoặc vấn đề khẩn cấp cần được giải quyết ngay tức thì."}},\n'
    ']'
    "\n\n"
    '### Examples:\n'
    '------------------------\n'
    '<history>\n'
    'alo\n'
    'vâng em đây\n'
    '</history>\n\n'
    '<input>\n'
    'cổng điện lực nhận hộ em cái đơn hàng hai triệu ba chị nhá\n'
    '</input>\n\n'
    '<output>\n'
    '{{"intention": [{{"predicted_labels": ["Lấy thông tin COD Lấy", "KN - Shop đã nhận trả hàng", "Hẹn lấy"]}}]}}\n'
    '</output>\n'
    '------------------------\n'
    '<history>\n'
    'alo em bên giao hàng chị ơi mình có đơn hàng gửi về chỗ thủy lợi hai này\n'
    'bao tiền đấy anh\n'
    'à đơn hàng của chị nó ghi là một trăm lẻ ba chị ạ\n'
    'à nó ghi là hôi nách à về sữa tắm à\n'
    '</history>\n\n'
    '<input>\n'
    'đoạn nào thủy lợi đấy chị ơi để em giao hàng\n'
    '</input>\n\n'
    '<output>\n'
    '{{"intention": [{{"predicted_labels": ["Thông tin địa chỉ ĐH - Vị trí hiện tại"]}}]}}\n'
    '</output>\n'
    '------------------------\n\n'
    "### Xử lý input sau\n"
    "<input>\n"
    "{text}\n"
    "</input>\n"
)