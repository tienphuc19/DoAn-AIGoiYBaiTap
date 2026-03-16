from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine # Thêm thư viện này
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập - Cloud Server Edition")

# ==========================================
# 1. CẤU HÌNH KẾT NỐI DATABASE CHO CLOUD SERVER (LINUX)
# ==========================================
def get_db_connection():
    # Sử dụng pymssql thay cho pyodbc để chạy mượt trên Linux server
    db_url = "mssql+pymssql://userPersonalizedSystem:123456789@118.69.126.49/Data_PersonalizedSystem"
    engine = create_engine(db_url)
    return engine.connect()

# (Các phần load_data_from_sql và API bên dưới BẠN GIỮ NGUYÊN HOÀN TOÀN NHÉ)

def load_data_from_sql():
    try:
        conn = get_db_connection()
        
        # Đọc dữ liệu bài tập
        query_exercises = """
            SELECT 
                Id AS ExerciseID, 
                TenBaiTap AS Title, 
                ISNULL(MaMon, '') + ' ' + TenBaiTap AS Tags,
                ISNULL(MaDoKho, 1) AS Difficulty
            FROM BAITAP
        """
        df_exercises = pd.read_sql(query_exercises, conn)
        
        # Đọc lịch sử làm bài (ĐÃ ĐỔI THÀNH ĐIỂM SỐ)
        query_history = """
            SELECT 
                MaSinhVien AS StudentID, 
                MaBaiTap AS ExerciseID, 
                DiemSo AS Score 
            FROM AI_LichSuLamBai
        """
        df_history = pd.read_sql(query_history, conn)
        
        conn.close()
        return df_exercises, df_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy xuất dữ liệu Remote SQL: {str(e)}")

# ==========================================
# 2. LUỒNG SINH VIÊN: AI GỢI Ý BÀI TẬP 
# ==========================================
class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 3

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations(
    request: RecommendRequest, 
    x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")
):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập! Chỉ sinh viên mới được nhận gợi ý.")

    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    
    # AI ĐÁNH GIÁ: Nếu điểm >= 5.0 thì coi như đã vượt qua bài đó
    passed_exercises = df_history[(df_history['StudentID'] == request.student_id) & (df_history['Score'] >= 5.0)]['ExerciseID'].tolist()
    
    # Xác định Năng lực hiện tại
    if not passed_exercises:
        easy_exercises = df_exercises[df_exercises['Difficulty'] == 1].head(request.top_k).to_dict('records')
        return {"status": "new_student", "current_level": 1, "message": "Gợi ý cơ bản", "recommendations": easy_exercises}
    else:
        current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises)]['Difficulty'].max()
        if pd.isna(current_level): current_level = 1

    # Huấn luyện thuật toán
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises)].index.tolist()
    sim_scores = sum([cosine_sim[i] for i in indices])
    
    related_indices = sim_scores.argsort()[::-1]
    recommendations = []
    
    for idx in related_indices:
        row = df_exercises.iloc[idx]
        ex_id = row['ExerciseID']
        ex_diff = row['Difficulty']
        
        if ex_id not in passed_exercises and ex_diff <= current_level + 1:
            recommendations.append(row.to_dict())
            
        if len(recommendations) >= request.top_k: 
            break

    return {
        "status": "success", 
        "current_level": int(current_level),
        "recommendations": recommendations
    }

# ==========================================
# 3. API NỘP BÀI TẬP (NHẬN ĐIỂM SỐ 0-10)
# ==========================================
class SubmitResultRequest(BaseModel):
    student_id: int
    exercise_id: int
    score: float  # Nhận điểm số thập phân (VD: 8.5)

@app.post("/api/submit-result", tags=["Sinh Viên"])
def submit_exercise_result(
    request: SubmitResultRequest,
    x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")
):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
    
    # Chặn không cho nhập điểm vớ vẩn
    if request.score < 0 or request.score > 10:
        raise HTTPException(status_code=400, detail="Điểm số phải nằm trong khoảng từ 0 đến 10")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Lưu điểm số thẳng vào Database
        insert_query = "INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (request.student_id, request.exercise_id, request.score))
        conn.commit() 
        conn.close()
        return {"status": "success", "message": f"Đã lưu thành công {request.score} điểm cho bài tập {request.exercise_id}!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu Database: {str(e)}")

# ==========================================
# 4. LUỒNG GIẢNG VIÊN (Thống kê theo Điểm)
# ==========================================
@app.get("/api/teacher/student/{student_id}", tags=["Giảng Viên"])
def get_student_profile(
    student_id: int, 
    x_user_role: str = Header(None, description="Bắt buộc nhập 'teacher'")
):
    if x_user_role != "teacher":
        raise HTTPException(status_code=403, detail="Chỉ giảng viên mới xem được.")

    df_exercises, df_history = load_data_from_sql()
    student_data = df_history[df_history['StudentID'] == student_id]
    
    if student_data.empty:
        return {"status": "error", "message": "Sinh viên chưa làm bài."}

    # Phân loại: >= 5.0 là Giỏi/Khá, < 5.0 là Yếu
    passed_ex = student_data[student_data['Score'] >= 5.0]['ExerciseID'].tolist()
    failed_ex = student_data[student_data['Score'] < 5.0]['ExerciseID'].tolist()

    strong_tags = df_exercises[df_exercises['ExerciseID'].isin(passed_ex)]['Tags'].explode().unique().tolist()
    weak_tags = df_exercises[df_exercises['ExerciseID'].isin(failed_ex)]['Tags'].explode().unique().tolist()
    weak_tags = [tag for tag in weak_tags if tag not in strong_tags]

    return {
        "student_id": student_id,
        "total_attempts": len(student_data),
        "good_score_count": len(passed_ex),
        "low_score_count": len(failed_ex),
        "strong_skills": strong_tags,
        "weak_skills": weak_tags
    }
    # ==========================================
# ==========================================
# 5. GIAO DIỆN DEMO V2 (TÍCH HỢP GIẢNG VIÊN & CLICK BÀI TẬP)
# ==========================================
@app.get("/demo", tags=["Giao Diện Demo"], response_class=HTMLResponse)
def get_demo_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Demo Hệ Thống AI</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #eef2f3; padding: 20px; }
            .container { max-width: 700px; margin: auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            /* Style cho Tabs */
            .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }
            .tab { padding: 10px 20px; cursor: pointer; font-weight: bold; color: #555; border-bottom: 3px solid transparent; }
            .tab.active { color: #2196F3; border-bottom: 3px solid #2196F3; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            
            /* Style nội dung */
            .card { border-left: 5px solid #4CAF50; background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; cursor: pointer; transition: 0.3s; }
            .card:hover { transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
            .level { display: inline-block; padding: 4px 10px; background: #ff9800; color: white; border-radius: 15px; font-size: 12px; font-weight: bold; }
            input, button { padding: 10px; margin-top: 10px; border-radius: 5px; border: 1px solid #ccc; }
            button { background: #2196F3; color: white; cursor: pointer; border: none; font-weight: bold; }
            button:hover { background: #0b7dda; }
            
            /* Cục thống kê Giảng viên */
            .stat-box { background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px; }
            .tag-pill { display: inline-block; background: #4CAF50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; margin: 3px; }
            .tag-pill.weak { background: #f44336; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="text-align: center; color: #333;">🤖 HỆ THỐNG TRÍ TUỆ NHÂN TẠO</h2>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('student')">👨‍🎓 Góc nhìn Sinh Viên</div>
                <div class="tab" onclick="switchTab('teacher')">👨‍🏫 Góc nhìn Giảng Viên</div>
            </div>

            <div id="student" class="tab-content active">
                <p>Nhập Mã SV để AI phân tích lộ trình học:</p>
                <input type="number" id="studentId" placeholder="Mã SV (VD: 1, 2...)" value="1">
                <button onclick="getRecommendations()">Xem Lộ Trình Của Tôi</button>
                <p style="font-size: 12px; color: #888;">* Gợi ý: Click vào bài tập để giả lập làm bài và chấm điểm.</p>
                <div id="result" style="margin-top: 20px;"></div>
            </div>

            <div id="teacher" class="tab-content">
                <p>Nhập Mã SV để xem báo cáo năng lực:</p>
                <input type="number" id="teacherStudentId" placeholder="Mã SV (VD: 1, 2...)" value="1">
                <button onclick="getTeacherStats()" style="background: #9c27b0;">Phân Tích Năng Lực</button>
                <div id="teacherResult" style="margin-top: 20px;"></div>
            </div>
        </div>

        <script>
            // Đổi Tab
            function switchTab(tabId) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }

            // GAPI Sinh Viên
            async function getRecommendations() {
                const stuId = document.getElementById('studentId').value;
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = "<i>Đang kết nối AI...</i>";

                try {
                    const response = await fetch('/api/recommend', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'x-user-role': 'student' },
                        body: JSON.stringify({ student_id: parseInt(stuId), top_k: 3 })
                    });
                    const data = await response.json();
                    
                    if(data.recommendations) {
                        resultDiv.innerHTML = `<h3>🎯 Lộ trình đề xuất (Năng lực Level: ${data.current_level})</h3>`;
                        data.recommendations.forEach(item => {
                            resultDiv.innerHTML += `
                                <div class="card" onclick="simulateExercise(${stuId}, ${item.ExerciseID}, '${item.Title}')">
                                    <h4 style="margin: 0 0 10px 0;">ID: ${item.ExerciseID} - ${item.Title} 🖱️</h4>
                                    <p style="margin: 0 0 10px 0; font-size: 14px; color: #555;">📚 Nguồn bài: ${item.Tags}</p>
                                    <span class="level">Độ khó: Level ${item.Difficulty}</span>
                                </div>
                            `;
                        });
                    } else {
                        resultDiv.innerHTML = "Không tìm thấy dữ liệu!";
                    }
                } catch (error) {
                    resultDiv.innerHTML = "<span style='color:red'>Lỗi Server.</span>";
                }
            }

            // Gợi ý làm bài (Simulate Submission)
            async function simulateExercise(stuId, exId, title) {
                let score = prompt(`Giả lập nộp bài cho ID ${exId} - ${title}\\nNhập điểm số đạt được (0 - 10):`);
                if (score !== null && score !== "") {
                    score = parseFloat(score);
                    if(score >= 0 && score <= 10) {
                        try {
                            const response = await fetch('/api/submit-result', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json', 'x-user-role': 'student' },
                                body: JSON.stringify({ student_id: stuId, exercise_id: exId, score: score })
                            });
                            const data = await response.json();
                            alert(data.message + "\\n\\nHãy bấm 'Xem Lộ Trình Của Tôi' lần nữa để xem AI thay đổi gợi ý nhé!");
                        } catch (e) {
                            alert("Lỗi khi lưu điểm!");
                        }
                    } else {
                        alert("Điểm phải từ 0 đến 10!");
                    }
                }
            }

            // GAPI Giảng Viên
            async function getTeacherStats() {
                const stuId = document.getElementById('teacherStudentId').value;
                const resultDiv = document.getElementById('teacherResult');
                resultDiv.innerHTML = "<i>Đang trích xuất dữ liệu...</i>";

                try {
                    const response = await fetch(`/api/teacher/student/${stuId}`, {
                        headers: { 'x-user-role': 'teacher' }
                    });
                    const data = await response.json();
                    
                    if(data.status === "error") {
                        resultDiv.innerHTML = `<span style='color:red'>${data.message}</span>`;
                        return;
                    }

                    let strongTagsHtml = data.strong_skills.map(t => `<span class="tag-pill">${t}</span>`).join('');
                    let weakTagsHtml = data.weak_skills.map(t => `<span class="tag-pill weak">${t}</span>`).join('');

                    resultDiv.innerHTML = `
                        <div class="stat-box">
                            <h3 style="margin-top:0;">📊 Báo cáo Sinh viên ID: ${data.student_id}</h3>
                            <p><b>Tổng số bài đã làm:</b> ${data.total_attempts}</p>
                            <p><b>Số bài điểm cao (>=5.0):</b> <span style="color:green; font-weight:bold;">${data.good_score_count}</span></p>
                            <p><b>Số bài điểm thấp (<5.0):</b> <span style="color:red; font-weight:bold;">${data.low_score_count}</span></p>
                            
                            <hr style="border:0; border-top:1px solid #ccc; margin: 15px 0;">
                            
                            <h4>💪 Kỹ năng thế mạnh (Đã qua bài):</h4>
                            <div>${strongTagsHtml || '<i>Chưa có dữ liệu</i>'}</div>
                            
                            <h4 style="margin-top: 15px;">⚠️ Kiến thức bị hổng (Điểm thấp):</h4>
                            <div>${weakTagsHtml || '<i>Chưa có dữ liệu</i>'}</div>
                        </div>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = "<span style='color:red'>Lỗi hệ thống.</span>";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)