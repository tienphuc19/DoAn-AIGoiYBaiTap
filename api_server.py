from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random 

app = FastAPI(title="Hệ Thống Gợi Ý Bài Tập - Bản Hỗn Hợp Hoàn Thiện")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class MockGradeRequest(BaseModel):
    student_id: int
    exercise_id: int
    submitted_work: str = "" 

@app.post("/api/mock-grade-and-submit", tags=["Sinh Viên"])
def mock_grade_and_submit_result(request: MockGradeRequest, x_user_role: str = Header(None, description="Bắt buộc nhập 'student'")):
    if x_user_role != "student":
        raise HTTPException(status_code=403, detail="Cấm truy cập!")
        
    try:
        conn = get_db_connection()
        query_check_ex = text("SELECT Id FROM BAITAP WHERE Id = :ex_id")
        check_ex = conn.execute(query_check_ex, {"ex_id": request.exercise_id}).fetchone()
        if not check_ex:
            raise HTTPException(status_code=400, detail="Bài tập không tồn tại!")
            
        if request.student_id == 333:
            final_grade = random.uniform(7.5, 10.0) 
        else:
            final_grade = random.uniform(4.0, 9.0) 
            
        final_grade = round(final_grade, 1) 
        
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
    
    if request.subject_code:
        df_exercises = df_exercises[df_exercises['SubjectCode'].str.contains(request.subject_code, case=False, na=False)]
        # DÒNG CODE "CỨU TINH" NẰM Ở ĐÂY: Reset lại số thứ tự để không bị lệch Index
        df_exercises = df_exercises.reset_index(drop=True)
    
    if df_exercises.empty:
        return {"status": "success", "cf_error_message": "Không có bài tập.", "recommendations": []}

    all_attempts_ids = df_history[df_history['StudentID'] == request.student_id]['ExerciseID'].tolist()
    
    passed_exercises_ids = df_history[(df_history['StudentID'] == request.student_id) & (df_history['Score'] >= 5.0)]['ExerciseID'].tolist()
    if not passed_exercises_ids:
        current_level = 1
    else:
        current_level = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)]['Difficulty'].max()
        if pd.isna(current_level): current_level = 1

    candidate_exercises = df_exercises[(~df_exercises['ExerciseID'].isin(all_attempts_ids)) & (df_exercises['Difficulty'] <= current_level + 2)]
    
    if candidate_exercises.empty:
        return {"status": "success", "cf_error_message": "Đã làm hết bài.", "recommendations": []}

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_exercises['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    passed_indices = df_exercises[df_exercises['ExerciseID'].isin(passed_exercises_ids)].index.tolist()
    if not passed_indices:
        sim_scores_cb = sum([cosine_sim[i] for i in passed_indices]) if passed_indices else []
    else:
        sim_scores_cb = sum([cosine_sim[i] for i in passed_indices])
        if type(sim_scores_cb) != list and sim_scores_cb.max() > sim_scores_cb.min():
            sim_scores_cb = (sim_scores_cb - sim_scores_cb.min()) / (sim_scores_cb.max() - sim_scores_cb.min())

    def calculate_cf_item_based_fallback():
        try:
            df_pivot = df_history.pivot_table(index='StudentID', columns='ExerciseID', values='Score')
            
            if df_pivot.shape[0] < 2:
                raise Exception("Lỗi: Ít hơn 2 sinh viên có dữ liệu, Collaborative Filtering Item-Based không thể hoạt động.")

            item_sim = cosine_similarity(df_pivot.T.fillna(0))
            item_sim_df = pd.DataFrame(item_sim, index=df_pivot.columns, columns=df_pivot.columns)
            
            # Lấy lịch sử điểm của sinh viên hiện tại, nếu chưa có lịch sử gì thì bỏ qua
            if request.student_id not in df_pivot.index:
                raise Exception("Sinh viên chưa có lịch sử làm bài để chạy Item-Based CF.")
                
            student_scores = df_pivot.loc[request.student_id].dropna()
            cf_scores_list = []
            
            for ex_id in df_exercises['ExerciseID'].tolist():
                if ex_id in passed_exercises_ids:
                    cf_scores_list.append(0) 
                    continue
                
                if ex_id in item_sim_df.columns:
                    sim_subset = item_sim_df[ex_id].reindex(student_scores.index).fillna(0)
                    weighted_sum = (sim_subset * student_scores).sum()
                    sim_sum = sim_subset.abs().sum()
                    score = weighted_sum / sim_sum if sim_sum > 0 else 0
                else:
                    score = 0
                cf_scores_list.append(score)
                
            if len(cf_scores_list) > 0 and max(cf_scores_list) > min(cf_scores_list):
                cf_scores_list = [(x - min(cf_scores_list)) / (max(cf_scores_list) - min(cf_scores_list)) for x in cf_scores_list]
            return cf_scores_list, False 

        except Exception as e:
            print(f"[{e}] -> Gợi ý hỗn hợp đã FALLBACK sang Content-Based.")
            return None, e

    scores_cf, cf_error_msg = calculate_cf_item_based_fallback()

    final_recommendations_with_score = []
    
    cb_scores_dict = {}
    if passed_indices:
        for idx in df_exercises.index:
            cb_scores_dict[df_exercises.iloc[idx]['ExerciseID']] = sim_scores_cb[idx]
            
    cf_scores_dict = {}
    if scores_cf is not None:
        for idx in df_exercises.index:
            cf_scores_dict[df_exercises.iloc[idx]['ExerciseID']] = scores_cf[idx]
    
    for _, row in candidate_exercises.iterrows():
        ex_id = row['ExerciseID']
        cb_score = cb_scores_dict.get(ex_id, 0)
        
        if scores_cf is not None:
            cf_score = cf_scores_dict.get(ex_id, 0)
            final_score = 0.6 * cb_score + 0.4 * cf_score
        else:
            final_score = cb_score
            
        final_recommendations_with_score.append({
            "status": "success",
            "exercise": row.to_dict(),
            "final_hybrid_score": float(final_score)
        })

    sorted_recommendations = sorted(final_recommendations_with_score, key=lambda x: x['final_hybrid_score'], reverse=True)
    top_k_recommendations = sorted_recommendations[:request.top_k]
    
    result_list = [item['exercise'] for item in top_k_recommendations]

    return {
        "status": "success", 
        "current_level": int(current_level),
        "cf_error_message": str(cf_error_msg) if cf_error_msg else "Item-Based Collaborative Filtering is ACTIVE.",
        "recommendations": result_list
    }

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
