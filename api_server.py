from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random # Dùng cho Mock AI Grading

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập - Bản Hỗn Hợp Hoàn Thiện")

# ==========================================
# CẤP PHÉP BẢO MẬT CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. CẤU HÌNH & TRUY XUẤT DATABASE
# ==========================================
def get_db_connection():
    db_url = "mssql+pymssql://userPersonalizedSystem:123456789@118.69.126.49/Data_PersonalizedSystem"
    engine = create_engine(db_url)
    return engine.connect()

def load_data_from_sql():
    try:
        conn = get_db_connection()
        query_exercises = """
            SELECT 
                Id AS ExerciseID, 
                TenBaiTap AS Title, 
                ISNULL(MaMon, '') AS SubjectCode,
                ISNULL(MaMon, '') + ' ' + ISNULL(TenBaiTap, '') AS Tags,
                ISNULL(MaDoKho, 1) AS Difficulty
            FROM BAITAP
        """
        df_exercises = pd.read_sql(query_exercises, conn)
        
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
# 2. MOCK AI GRADING & TIÊU CHÍ (API CHẤM ĐIỂM)
# ==========================================
class MockGradeRequest(BaseModel):
    student_id: int
    exercise_id: int
    submitted_work: str = "" # Dữ liệu làm bài từ frontend gửi lên

@app.post("/api/mock-grade-and-submit", tags=["Sinh Viên"])
def mock_grade_and_submit_result(request: MockGradeRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
        
    try:
        conn = get_db_connection()
        
        # 1. Lấy thông tin & tiêu chí bài tập từ SQL
        # Bảng TIÊU CHÍ bạn gửi không có trường điểm rõ ràng, nên tôi sẽ mock điểm ngẫu nhiên
        # trong khoảng từ 4-10 để mô phỏng điểm số dựa trên tiêu chí.
        # Khi đồ án thật, đây sẽ là nơi bạn tích hợp mô hình AI chấm điểm.
        query_check_ex = text("SELECT Id FROM BAITAP WHERE Id = :ex_id")
        check_ex = conn.execute(query_check_ex, {"ex_id": request.exercise_id}).fetchone()
        if not check_ex:
            raise HTTPException(status_code=400, detail="Bài tập không tồn tại!")
            
        # 2. Mock AI Chấm Điểm (Làm thành kịch bản)
        # Sinh viên Tienphuc (333) làm tốt hơn, Sinh viên01 (111) làm kém hơn
        if request.student_id == 333:
            final_grade = random.uniform(7.5, 10.0) # Tienphuc làm tốt
        else:
            final_grade = random.uniform(4.0, 9.0) # Sinh viên01 làm ngẫu nhiên
            
        final_grade = round(final_grade, 1) # Làm tròn 1 chữ số thập phân
        
        # 3. Lưu điểm vào database
        # Chúng ta dùng "upsert" logic: Nếu đã làm thì cập nhật, chưa làm thì thêm mới
        query_upsert = text("""
            IF EXISTS (SELECT 1 FROM AI_LichSuLamBai WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id)
                UPDATE AI_LichSuLamBai SET DiemSo = :grade WHERE MaSinhVien = :sv_id AND MaBaiTap = :ex_id
            ELSE
                INSERT INTO AI_LichSuLamBai (MaSinhVien, MaBaiTap, DiemSo) VALUES (:sv_id, :ex_id, :grade)
        """)
        conn.execute(query_upsert, {"sv_id": request.student_id, "ex_id": request.exercise_id, "grade": final_grade})
        conn.commit() 
        conn.close()
        
        return {
            "status": "success", 
            "score": final_grade,
            "passed": final_grade >= 5.0,
            "message": f"🤖 AI Chấm Điểm dựa trên tiêu chí: Bạn đạt {final_grade} điểm và đã { 'vượt qua' if final_grade >= 5.0 else 'chưa vượt qua' } bài tập này. Điểm đã được lưu thành công!"
        }
    except HTTPException as e: raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi chấm điểm: {str(e)}")

# ==========================================
# 3. THUẬT TOÁN GỢI Ý HỖN HỢP (HYBRID - KT + CBF + CF)
# ==========================================
# (Hỗn hợp: Knowledge Tracing phỏng đoán level, Content-Based gợi ý chủ đề, 
# Item-Based CF mở rộng dựa trên các sinh viên khác)
class RecommendRequest(BaseModel):
    student_id: int
    top_k: int = 6
    subject_code: str = ""

@app.post("/api/recommend", tags=["Sinh Viên"])
def get_recommendations_hybrid(request: RecommendRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")

    df_exercises, df_history = load_data_from_sql()
    df_exercises['Tags'] = df_exercises['Tags'].fillna('')
    
    # LỌC LẠI: Chỉ lấy các bài thuộc đúng môn học được chọn
    if request.subject_code:
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(request.subject_code, case=False, na=False)]
    
    if df_exercises.empty:
        return {"status": "success", "recommendations": []}

    # Lấy lịch sử làm bài sinh viên (Cả qua và không qua)
    all_attempts_ids = df_history[df_history['StudentID'] == request.student_id]['ExerciseID'].tolist()
    
    # 3.1. KNOWLEDGE TRACING (Phỏng đoán năng lực - Nâng cấp)
    passed_exercises_ids = df_history[(df_history['StudentID'] == request.student_id) & (df_history['Score'] >= 5.0)]['ExerciseID'].tolist()
    if not passed_exercises_ids:
        current_level = 1
    else:
        current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)]['Difficulty'].max()
        if pd.isna(current_level): current_level = 1

    # Lọc bài tập: Chỉ gợi ý bài có độ khó Level + 2 và chưa làm qua
    candidate_exercises = df_exercises[(~df_exercises['ExerciseID'].isin(all_attempts_ids)) & (df_exercises['Difficulty'] <= current_level + 2)]
    
    if candidate_exercises.empty:
        return {"status": "success", "recommendations": []}

    # 3.2. CONTENT-BASED FILTERING (Dựa trên chủ đề - 60% trọng số)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Lấy độ tương đồng cosine cho sinh viên dựa trên các bài đã vượt qua
    passed_indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)].index.tolist()
    if not passed_indices:
        sim_scores_cb = sum([cosine_sim[i] for i in passed_indices]) if passed_indices else []
    else:
        sim_scores_cb = sum([cosine_sim[i] for i in passed_indices])
        # Chuẩn hóa CB scores về 0-1
        sim_scores_cb = (sim_scores_cb - sim_scores_cb.min()) / (sim_scores_cb.max() - sim_scores_cb.min()) if sim_scores_cb.max() > sim_scores_cb.min() else sim_scores_cb

    # 3.3. ITEM-BASED COLLABORATIVE FILTERING (40% trọng số - FALLBACK)
    # Chúng ta xây dựng matrix cho Item-Based CF, nhưng với dữ liệu SQL hiện tại chỉ có 1 sinh viên, 
    # thuật toán này sẽ trả về lỗi hoặc fallback về Content-Based.
    def calculate_cf_item_based_fallback():
        try:
            # Tạo Matrix SinhVien - BaiTap
            df_pivot = df_history.pivot_table(index='StudentID', columns='ExerciseID', values='Score')
            
            # Cảnh báo lỗi: Ít sinh viên làm Item-Based CF (Cần ít nhất 2 sinh viên làm bài)
            if df_pivot.shape[0] < 2:
                raise Exception("Lỗi: Ít hơn 2 sinh viên có dữ liệu, Collaborative Filtering Item-Based không thể hoạt động. Hệ thống sẽ tạm thời tắt Item-Based CF và chuyển sang 100% Content-Based.")

            # Tính tương đồng Item-Based Cosine Similarity cho các bài tập
            item_sim = cosine_similarity(df_pivot.T.fillna(0))
            item_sim_df = pd.DataFrame(item_sim, index=df_pivot.columns, columns=df_pivot.columns)
            
            # Tính điểm phỏng đoán dựa trên lịch sử
            student_scores = df_pivot.loc[request.student_id].dropna()
            cf_scores_list = []
            
            for ex_id in df_exercises['ExerciseID'].tolist():
                if ex_id in passed_exercises_ids:
                    cf_scores_list.append(0) # Đã làm
                    continue
                
                # Tính tổng weighted similarity
                sim_subset = item_sim_df[ex_id].reindex(student_scores.index)
                weighted_sum = (sim_subset * student_scores).sum()
                sim_sum = sim_subset.abs().sum()
                
                score = weighted_sum / sim_sum if sim_sum > 0 else 0
                cf_scores_list.append(score)
                
            # Chuẩn hóa CF scores về 0-1
            cf_scores_list = (cf_scores_list - min(cf_scores_list)) / (max(cf_scores_list) - min(cf_scores_list)) if max(cf_scores_list) > min(cf_scores_list) else cf_scores_list
            return cf_scores_list, False # Hoạt động bình thường

        except Exception as e:
            # Fallback: Collaborative Filtering không chạy
            print(f"[{e}] -> Gợi ý hỗn hợp đã FALLBACK sang Content-Based.")
            return None, e

    # Gọi Item-Based CF
    scores_cf, cf_error_msg = calculate_cf_item_based_fallback()

    # 3.4. KẾT HỢP HỖN HỢP (HYBRID)
    final_recommendations_with_score = []
    
    # Chuyển sim_scores_cb sang một dict để tra cứu cho nhanh
    cb_scores_dict = {}
    if passed_indices:
        for idx in df_exercises.index:
            cb_scores_dict[df_exercises.iloc[idx]['ExerciseID']] = sim_scores_cb[idx]
            
    # Chuyển sim_scores_cf sang một dict để tra cứu cho nhanh
    cf_scores_dict = {}
    if scores_cf is not None:
        for idx in df_exercises.index:
            cf_scores_dict[df_exercises.iloc[idx]['ExerciseID']] = scores_cf[idx]
    
    # Tính điểm final hỗn hợp
    # 0.6 Content-Based + 0.4 Item-Based CF (Nếu hoạt động)
    for _, row in candidate_exercises.iterrows():
        ex_id = row['ExerciseID']
        
        # Nếu CB not found (chưa học bài nào), cho điểm 0
        cb_score = cb_scores_dict.get(ex_id, 0)
        
        if scores_cf is not None:
            cf_score = cf_scores_dict.get(ex_id, 0)
            # Điểm hỗn hợp
            final_score = 0.6 * cb_score + 0.4 * cf_score
        else:
            # Fallback: 100% Content-Based
            final_score = cb_score
            
        final_recommendations_with_score.append({
            "status": "success",
            "exercise": row.to_dict(),
            "cb_score": float(cb_score),
            "cf_score": float(scores_cf[idx] if scores_cf is not None else 0),
            "final_hybrid_score": float(final_score)
        })

    # Sắp xếp và lấy top K
    sorted_recommendations = sorted(final_recommendations_with_score, key=lambda x: x['final_hybrid_score'], reverse=True)
    top_k_recommendations = sorted_recommendations[:request.top_k]
    
    # Trả về kết quả sạch (chỉ lấy dictionary bài tập)
    result_list = [item['exercise'] for item in top_k_recommendations]

    # Trả về kèm thông báo lỗi của Collaborative Filtering để sinh viên biết lý do thuật toán tắt
    return {
        "status": "success", 
        "current_level": int(current_level),
        "cf_error_message": str(cf_error_msg) if cf_error_msg else "Item-Based Collaborative Filtering is ACTIVE.",
        "recommendations": result_list
    }

# ==========================================
# 4. API ĐĂNG NHẬP & GIAO DIỆN
# ==========================================
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login", tags=["Hệ Thống"])
def login_user(request: LoginRequest):
    try:
        conn = get_db_connection()
        query = text("SELECT VaiTro, MaNguoiDung, HoTen FROM TAIKHOAN WHERE TenDangNhap = :usr AND MatKhau = :pwd")
        df_user = pd.read_sql(query, conn, params={"usr": request.username, "pwd": request.password})
        conn.close()

        if df_user.empty:
            return {"status": "error", "message": "Sai tên đăng nhập hoặc mật khẩu!"}
        
        user_info = df_user.iloc[0]
        return {
            "status": "success",
            "role": user_info['VaiTro'],
            "user_id": int(user_info['MaNguoiDung']),
            "full_name": user_info['HoTen']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Database: {str(e)}")

@app.get("/", tags=["Hệ Thống"])
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Vui lòng upload index.html lên GitHub!"}
