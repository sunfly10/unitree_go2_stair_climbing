#캘리브레이션을 위한 데이터 생성 코드:

import open3d as o3d
import numpy as np
import copy
import math

def get_rotation_matrix(r_deg, p_deg, y_deg):
    r = math.radians(r_deg)
    p = math.radians(p_deg)
    y = math.radians(y_deg)
    return o3d.geometry.get_rotation_matrix_from_xyz([r, p, y])

def main():
    print("데이터 불러오는 중...")
    try:
        target = o3d.io.read_point_cloud("depth_data.pcd") 
        source = o3d.io.read_point_cloud("lidar_data.pcd")
    except:
        print("파일이 없습니다.")
        return

    # 보기 편하게 다운샘플링 및 색상 지정
    target = target.voxel_down_sample(0.05)
    source = source.voxel_down_sample(0.05)
    target.paint_uniform_color([0, 0, 1]) # 파랑 (카메라/기준)
    source.paint_uniform_color([1, 0, 0]) # 빨강 (라이다/회전시킬 것)

    # -----------------------------------------------------
    # 테스트할 후보군 (가장 많이 쓰이는 조합 4개)
    # -----------------------------------------------------
    candidates = [
        {"id": 1, "R": -90, "P": 0, "Y": -90}, # 현재 설정 (실패한 것으로 보임)
        {"id": 2, "R": 90,  "P": 0, "Y": -90}, # Roll 반대
        {"id": 3, "R": 0,   "P": 90, "Y": 0},  # Pitch로 세우기
        {"id": 4, "R": -90, "P": -90, "Y": 0}, # 복합 회전
        {"id": 5, "R": 0,   "P": 0,   "Y": 0}, # 회전 없음 (순정)
    ]

    print("\n[후보군 생성 시작]")
    for c in candidates:
        source_temp = copy.deepcopy(source)
        R_mat = get_rotation_matrix(c["R"], c["P"], c["Y"])
        
        # 회전 적용
        source_temp.rotate(R_mat, center=(0,0,0))
        
        # 파일 저장
        filename = f"check_rot_{c['id']}.pcd"
        o3d.io.write_point_cloud(filename, target + source_temp)
        print(f"👉 저장됨: {filename} (Roll={c['R']}, Pitch={c['P']}, Yaw={c['Y']})")

    print("\n" + "="*50)
    print("✅ 확인 방법:")
    print("터미널에서 'open3d check_rot_1.pcd', 'open3d check_rot_2.pcd' ... 하나씩 여세요.")
    print("빨간색 바닥과 파란색 바닥이 '평행하게(ㅡ자)' 겹쳐 보이는 번호를 알려주세요!")
    print("="*50)

if __name__ == '__main__': main()
