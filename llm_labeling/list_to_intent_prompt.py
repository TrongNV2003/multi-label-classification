import json

label_list = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]

json_data = [{"intent": label} for label in label_list]

output_lines = [f"'    {{\"intent\": \"{label}\"}},\\n'" for label in label_list]
output_str = "[\n" + "\n".join(output_lines) + "\n]"

print(output_str)

output_file = "intent_list_exact_syntax.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_str)
print(f"Đã lưu vào file: {output_file}")