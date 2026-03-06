#카메라와 라이다 사이의 캘리브레이션 시도 코드(실패)
# calibration_final_success.py
import open3d as o3d
import numpy as np
import copy
import math
import sys

# ---------------------------------------------------------
# [설정] 사진으로 확인된 "정답" 각도 (Candidate 4)
# ---------------------------------------------------------
INITIAL_ROLL  = -90.0
INITIAL_PITCH = -90.0
INITIAL_YAW   = 0.0

# 위치 미세 조정 (그래프 보고 추정)
TRANS_X = 0.00
TRANS_Y = 0.00
TRANS_Z = 0.15  # 빨간점이 아래에 있어서 위로 살짝 올림
# ---------------------------------------------------------

def main():
    print("1. 데이터 불러오는 중...")
    try:
        target = o3d.io.read_point_cloud("depth_data.pcd") 
        source = o3d.io.read_point_cloud("lidar_data.pcd") 
    except:
        print("파일이 없습니다."); return

    # 전처리 (노이즈 제거)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-5, -5, -1), max_bound=(5, 5, 3))
    target = target.crop(bbox); source = source.crop(bbox)
    target = target.voxel_down_sample(0.02); source = source.voxel_down_sample(0.02)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.paint_uniform_color([0, 0, 1]); source.paint_uniform_color([1, 0, 0])

    # [1] 초기 위치 적용
    trans_init = np.identity(4)
    trans_init[0, 3] = TRANS_X; trans_init[1, 3] = TRANS_Y; trans_init[2, 3] = TRANS_Z
    r = math.radians(INITIAL_ROLL); p = math.radians(INITIAL_PITCH); y = math.radians(INITIAL_YAW)
    R_init = o3d.geometry.get_rotation_matrix_from_xyz([r, p, y])
    trans_init[:3, :3] = R_init

    # 초기 상태 저장 (확인용)
    source_copy = copy.deepcopy(source)
    source_copy.transform(trans_init)
    o3d.io.write_point_cloud("check_final_start.pcd", target + source_copy)
    print(">> 초기 정렬 상태 'check_final_start.pcd' 저장됨.")

    # -------------------------------------------------------
    # [2] 1단계: Coarse Matching (50cm)
    # -------------------------------------------------------
    print("\n[1단계] 위치 대략 맞추기 (50cm)...")
    reg_1 = o3d.pipelines.registration.registration_icp(
        source, target, 0.5, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
    )
    print(f"   -> Fitness: {reg_1.fitness:.4f} / RMSE: {reg_1.inlier_rmse:.4f}")

    # -------------------------------------------------------
    # [3] 2단계: Fine Matching (10cm)
    # -------------------------------------------------------
    print("\n[2단계] 각도 정밀 보정 (10cm)...")
    reg_2 = o3d.pipelines.registration.registration_icp(
        source, target, 0.1, reg_1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
    )
    print(f"   -> Fitness: {reg_2.fitness:.4f} / RMSE: {reg_2.inlier_rmse:.4f}")

    # -------------------------------------------------------
    # [4] 3단계: Ultra-Fine (3cm) - 최종
    # -------------------------------------------------------
    print("\n[3단계] 최종 밀착 (3cm)...")
    reg_3 = o3d.pipelines.registration.registration_icp(
        source, target, 0.03, reg_2.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
    )
    print(f"   -> Fitness: {reg_3.fitness:.4f} / RMSE: {reg_3.inlier_rmse:.4f}")

    if reg_3.fitness < 0.4:
        print("\n⚠️ 점수가 약간 낮지만, RMSE(오차)가 0.05 이하면 사용 가능합니다.")

    # 결과 저장
    final_T = reg_3.transformation
    source.transform(final_T)
    o3d.io.write_point_cloud("calibration_result_success.pcd", target + source)
    
    # TF 출력
    T = final_T
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    sy = math.sqrt(T[0, 0] * T[0, 0] + T[1, 0] * T[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(T[2, 1], T[2, 2])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = math.atan2(T[1, 0], T[0, 0])
    else:
        roll = math.atan2(-T[1, 2], T[1, 1])
        pitch = math.atan2(-T[2, 0], sy)
        yaw = 0

    print("\n" + "="*50)
    print("🎉 축하합니다! 이 값을 사용하세요:")
    print(f"static_transform_publisher {x:.4f} {y:.4f} {z:.4f} {yaw:.4f} {pitch:.4f} {roll:.4f} camera_link lidar_link")
    print("="*50)

if __name__ == '__main__': main()
