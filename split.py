import cv2
import os
import numpy as np
import argparse

def process_dual_insv(video_front, video_back, output_dir, fps=2):
    os.makedirs(output_dir, exist_ok=True)
    cap_f = cv2.VideoCapture(video_front)
    cap_b = cv2.VideoCapture(video_back)
    
    if not cap_f.isOpened() or not cap_b.isOpened():
        print("错误：无法打开视频文件，请检查路径。")
        return

    frame_interval=10
    count = 0
    saved_count = 0
    
    print(f"正在从双视频中抽帧，每隔 {frame_interval} 帧抽取一次...")
    
    while True:
        ret_f, frame_f = cap_f.read()
        ret_b, frame_b = cap_b.read()
        if not ret_f or not ret_b:
            break
            
        if count % frame_interval == 0:
            h, w = frame_f.shape[:2]
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = int(min(w, h) * 0.47) 
            cv2.circle(mask, center, radius, (255, 255, 255), -1)
            
            frame_f_masked = cv2.rotate(cv2.bitwise_and(frame_f, mask), cv2.ROTATE_180)
            frame_b_masked = cv2.rotate(cv2.bitwise_and(frame_b, mask), cv2.ROTATE_180)

            cv2.imwrite(os.path.join(output_dir, f"frame_{saved_count:05d}_front.jpg"), frame_f_masked, [cv2.IMWRITE_JPEG_QUALITY, 98])
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved_count:05d}_back.jpg"), frame_b_masked, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"已提取 {saved_count} 对图像...")
            
        count += 1

    cap_f.release()
    cap_b.release()
    print(f"\n提取完成！共保存了 {saved_count * 2} 张鱼眼图到 {output_dir}")
    print(f"单张图片分辨率为: {w}x{h}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", "-f", required=True, help="前镜头视频路径 (包含 _00_ 的 insv)")
    parser.add_argument("--back", "-b", required=True, help="后镜头视频路径 (包含 _10_ 的 insv)")
    parser.add_argument("--output", "-o", required=True, help="输出 input 文件夹路径")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧率")
    args = parser.parse_args()
    
    process_dual_insv(args.front, args.back, args.output, args.fps)
