#include <chrono>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <cmath>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/exiter.hpp"
#include "tools/trajectory.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化工具类
  tools::Exiter exiter;
  tools::Plotter plotter;

  // 初始化io类
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化auto_aim类
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;
  
  while (!exiter.exit()) {
    // Your code start
    
    //读取相机图像和IMU数据
    camera.read(img, t);
    if (img.empty()) continue;
    
    //获取云台状态
    auto gimbal_state = gimbal.state();
    
    //设置IMU四元数到求解器
    q = gimbal.q(t);
    solver.set_R_gimbal2world(q);
    
    //YOLO检测装甲板
    auto armors = yolo.detect(img);
    
    if (!armors.empty()) {
      //使用第一个检测到的装甲板
      auto best_armor = armors.front();
      
      //求解装甲板3D位置
      solver.solve(best_armor);
      
      //使用世界坐标系计算目标角度
      Eigen::Vector3d target_pos = best_armor.xyz_in_world;
      
      //转换为球坐标系
      auto ypd = tools::xyz2ypd(target_pos);
      double target_yaw = ypd[0];
      double target_pitch = ypd[1];
      double distance = ypd[2];
      
      //弹道补偿计算
      double pitch_compensation = 0.0;
      if (gimbal_state.bullet_speed > 0) {
        // 计算水平距离和高度差
        double horizontal_distance = std::sqrt(target_pos[0]*target_pos[0] + target_pos[1]*target_pos[1]);
        double height_diff = target_pos[2];
        
        //轨迹计算功能要考虑重力影响，计算出子弹飞行时间
        tools::Trajectory trajectory(gimbal_state.bullet_speed, horizontal_distance, height_diff);
        if (!trajectory.unsolvable) {
          pitch_compensation = trajectory.pitch - target_pitch;
        }
      }
      
      //应用弹道补偿
      target_pitch += pitch_compensation;
      
      //反转pitch方向（修复第一次调试上下相反的问题）
      target_pitch = -target_pitch;
      
      // 修正系统偏差（第二次调试打在右下方，需要向左上偏移）
      // 向左偏移来修正右偏
      double yaw_offset = 0.06;
      // 向上偏移来修正下偏  
      double pitch_offset = -0.09;
      
      target_yaw += yaw_offset;
      target_pitch += pitch_offset;
      
      //判断是否瞄准装甲板
      double first_tolerance = 3.0 * CV_PI / 180.0;//deepseek给出的解决方式
      
      double yaw_error = std::abs(target_yaw - gimbal_state.yaw);
      double pitch_error = std::abs(target_pitch - gimbal_state.pitch);
      double total_error = std::sqrt(yaw_error * yaw_error + pitch_error * pitch_error);
      
      bool is_aimed = (total_error < first_tolerance);
      
      //发送云台控制命令
      gimbal.send(true, is_aimed, target_yaw, target_pitch);
      
      
      //输出Plotter数据
      nlohmann::json plot_data;
      plot_data["gimbal_yaw"] = target_yaw;
      plot_data["gimbal_pitch"] = target_pitch;
      plotter.plot(plot_data);
      
      cv::waitKey(1);
    }
    // Your code end
  }
  
  return 0;
}
