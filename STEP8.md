# Step 8 — Phân tích benchmark Memory Systems

## Cách đo

Kết quả dưới đây được tạo từ trạng thái sạch bằng:

```powershell
Remove-Item -Recurse -Force state
python src/benchmark.py
pytest src/test_agents.py -v
```

Benchmark luôn khởi tạo agent với `force_offline=True`, vì vậy số liệu không
phụ thuộc API key hoặc thời điểm chạy. `data/` được dùng nguyên trạng.

## Kết quả

| Suite | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 1,355 | 13,538 | 0.00 | 0.25 | 0 | 0 |
| Standard | Advanced | 5,408 | 34,876 | 1.00 | 1.00 | 228 | 0 |
| Long-Context Stress | Baseline | 233 | 21,950 | 0.00 | 0.25 | 0 | 0 |
| Long-Context Stress | Advanced | 648 | 10,819 | 1.00 | 1.00 | 182 | 11 |

## Bốn kết luận từ số liệu

### 1. Advanced có cross-session recall, Baseline thì không

Ở cả Standard và Stress, Baseline có recall `0.00` và memory growth `0` byte;
nó chỉ giữ `SessionState` theo `thread_id` và không hề ghi `User.md`. Recall
questions của benchmark dùng một thread mới, nên Baseline không thể dùng lại
lịch sử của thread trước.

Advanced đạt recall `1.00` ở cả hai suite, đồng thời ghi `228` byte (Standard)
và `182` byte (Stress). Mỗi message đi qua `extract_profile_updates()`, các
fact được `upsert_fact()` vào `User.md`, và `_offline_response()` đọc profile
này khi nhận câu hỏi ở thread mới. Persistent memory, thay vì summary, là lý
do chính cho recall xuyên phiên.

### 2. Advanced tốn hơn ở hội thoại bình thường

Trong Standard, Advanced xử lý `34,876` prompt tokens so với `13,538` của
Baseline, và sinh `5,408` tokens so với `1,355`. Mỗi lượt Advanced phải mang
`User.md`, summary, và các recent messages vào prompt. Offline responder cũng
trả về các bullet profile có cấu trúc, làm cột `Agent tokens only` tăng.

Đây là trade-off chấp nhận được ở hội thoại ngắn: compact chưa chạy (`0`), nên
chi phí duy trì memory chưa được bù bằng phần lịch sử bị loại bỏ.

### 3. Hội thoại dài làm chi phí context của Baseline tăng mạnh

Trong Stress, Baseline phải xử lý `21,950` prompt tokens trên 16 lượt dù chỉ
sinh `233` agent tokens. Điều này cho thấy chi phí chính đến từ việc mang lại
toàn bộ lịch sử thread ở từng lượt, không phải chỉ từ độ dài câu trả lời.

### 4. Compact chủ yếu tối ưu prompt tokens, không phải output tokens

Trong cùng Stress suite, Advanced chỉ xử lý `10,819` prompt tokens, thấp hơn
`11,131` token (xấp xỉ `51%`) so với Baseline, và `Compactions = 11` xác nhận
cơ chế đã kích hoạt. `CompactMemoryManager` thay phần message cũ bằng summary
và giữ hai message gần nhất theo cấu hình benchmark.

Ngược lại, `Agent tokens only` của Advanced vẫn là `648`, cao hơn `233` của
Baseline. Compact không nhằm tối ưu cột output này; nó tối ưu lượng context
được xử lý ở mỗi lượt. Đây cũng là lý do cần tách hai cột khi đọc benchmark.

## Tăng trưởng memory và rủi ro

`User.md` là lợi ích và cũng là chi phí: profile tăng dần theo các fact ổn
định, thể hiện ở `228` và `182` byte trong hai suite. Nếu không có guardrail,
file có thể phình lên hoặc giữ một fact sai sau câu nói nhiễu, khiến recall có
vẻ tốt nhưng không còn đúng.

## Bonus chọn: Conflict handling có cấu trúc

Bonus đã áp dụng là ghi fact theo khóa với `UserProfileStore.upsert_fact()`.
Một correction mới thay đúng dòng cũ, thay vì thêm thêm một location hoặc
profession mâu thuẫn. Extractor cũng từ chối các giá trị kiểu câu hỏi như
`"gì"`, `"hiện tại"`, hay `"đã thay đổi"`, tránh việc một recall question
ghi đè fact thật.

Nó giải quyết trực tiếp case đổi nơi ở trong fixture và góp phần giữ recall
`1.00` ở cả hai suite. Rủi ro còn lại là regex heuristic có thể bỏ sót cách
diễn đạt correction mới hoặc, trong một câu phức tạp, chọn sai fact cuối cùng.
Một bản production nên thêm confidence score, provenance/timestamp cho từng
fact, và chỉ cho phép cập nhật khi ngữ cảnh correction đủ rõ.

## Kiểm chứng

`pytest src/test_agents.py -v` pass 4 test độc lập với API key: read/write/edit
`User.md`, compact trigger, cross-session recall, và giảm prompt load ở thread
dài.
