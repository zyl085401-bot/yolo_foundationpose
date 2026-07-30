# 使用foundationpose的时候这个命令会显示分割物体轮廓形状的效果

python run_demo.py --vis_mode contour


python3 -m pip install pyrealsense2
python3 -m pip install -U "ultralytics>=8.3.0" --no-deps

python3 realtime_foundation/simulate_recorded_camera.py \
  --input realtime_foundation/outputs/error/refiner_coarse_160/frame_001172_register.npz \
  --config realtime_foundation/config.yaml

# 如需查看同一次 register 的完整候选筛选流水线，将 config.yaml 最底部的
# simulation.candidate_pipeline_debug_enabled 改为 true。结果保存在：
# outputs/simulated_results/<帧名>/candidate_pipeline/repeat_001/
# 其中 candidate_pipeline.json/npz 保存完整数据，各阶段 *_contact_sheet.jpg 用于人工检查。

export DISPLAY=:0
xhost +si:localuser:root

cd ~/workspace/yolo_foundationpose

bash FoundationPose/docker/run_container_jetson_ros2.sh \
  python3 realtime_foundation/run_realtime.py \
  --config realtime_foundation/config.yaml

# Jetson 性能关联采集：同时记录运行日志、50 ms GPU 状态、200 ms
# GPU/EMC/温度/功耗和 tegrastats。Ctrl+C 后自动生成 events.csv、
# correlated_events.csv 和 correlation_summary.json。
bash FoundationPose/docker/run_jetson_ros2_telemetry.sh

# 可选采样间隔（毫秒）
FULL_INTERVAL_MS=200 FAST_INTERVAL_MS=50 \
  bash FoundationPose/docker/run_jetson_ros2_telemetry.sh

# 初始化阶段的数据流与计时

# tracker 未初始化时，主线程直接等待新的 YOLO DetectionMessage。YOLO
# 发布候选目标后会唤醒主线程，并暂停提交下一帧推理；主线程立即使用消息
# 自带的同帧 RGB、depth、K 和 mask 执行 FoundationPose register。候选被
# 拒绝、register 失败或 init_only 完成并 reset 后，YOLO 恢复搜索。
#
# timing summary 中：
#   yolo_total                       YOLO 纯检测计算时间
#   foundation_total                 FoundationPose register 计算时间
#   yolo_foundation_compute_total    上述两项之和（纯模型计算总时间）
#   detection_consume_delay          YOLO 完成到主线程取得结果的通知/调度延迟
#   init_total                       YOLO 开始到初始化质量检查通过的端到端时间

  nvidia