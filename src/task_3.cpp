#include <chrono>
#include <opencv2/opencv.hpp>
#include <memory>
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/exiter.hpp"


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

  // 目标跟踪相关变量
  std::unique_ptr<auto_aim::Target> target;
  bool target_found = false;
  
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
      //选择优先级最高的装甲板作为目标
      auto best_armor = *std::min_element(armors.begin(), armors.end(),
        [](const auto_aim::Armor& a, const auto_aim::Armor& b) {
          return a.priority < b.priority;
        });
      
      //求解装甲板3D位置
      solver.solve(best_armor);
      
      //初始化或更新目标跟踪
      if (!target_found) {
        //初始化目标跟踪器
        Eigen::VectorXd P0_dig(11);
        //降低角速度初始方差，经过两次调试后，角速度初始方差为由10.0变为3.0
        P0_dig << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0;
        
        target = std::make_unique<auto_aim::Target>(
          best_armor, t, 0.1, 4, P0_dig);
        target_found = true;
      } else {
        // 更新目标跟踪
        target->predict(t);
        target->update(best_armor);
      }
      
      //检查目标是否收敛
      if (target->convergened()) {
        auto ekf_state = target->ekf_x();
        double angular_velocity = ekf_state[7]; //拟合出的角速度
        double current_angle = ekf_state[6];    
        
        //使用EKF预测的目标中心位置
        Eigen::Vector3d center_pos;
        center_pos[0] = ekf_state[0];
        center_pos[1] = ekf_state[2];
        center_pos[2] = ekf_state[4];
        
        //计算目标中心位置的球坐标
        auto ypd = tools::xyz2ypd(center_pos);
        double center_yaw = ypd[0];    
        double center_pitch = ypd[1];    
        double distance = ypd[2];
      
        // 基础延迟时间
        double base_delay = 0.02;  // 20ms基础延迟
        
        // 根据角速度动态调整延迟时间
        double angular_velocity_factor = std::abs(angular_velocity);
        double dynamic_delay;
        
        if (angular_velocity_factor >= 7.0) {
          // 高速：需要更长的预测时间
          dynamic_delay = 0.05;  // 50ms额外延迟
        } else if (angular_velocity_factor >= 4.0) {
          // 中速：中等预测时间
          dynamic_delay = 0.03;  // 30ms额外延迟
        } else {
          // 低速：较短预测时间
          dynamic_delay = 0.02;  // 20ms额外延迟
        }
        //命中情况
        //高速10中8
        //中速10中?
        //低速10中9
        
        double total_delay = base_delay + dynamic_delay;
        double system_delay = total_delay;
        //预测延迟后的角度
        double predicted_angle = current_angle + angular_velocity * system_delay;
        
        //计算预测角度与目标中心的角度差
        double angle_diff = std::abs(tools::limit_rad(predicted_angle - center_yaw));
        
        //角度阈值
        double abs_angular_velocity = std::abs(angular_velocity);
        double angle_threshold;
        if (abs_angular_velocity >= 7.0) {
          angle_threshold = 0.15;  
        } else if (abs_angular_velocity >= 4.0) {
          angle_threshold = 0.11;  
        } else {
          angle_threshold = 0.10;  
        }
        
        //判断是否开火
        bool is_aimed = (angle_diff < angle_threshold);
        
        //云台瞄准目标中心，保持静止（根据大作业tips）
        double target_yaw = center_yaw;
        double target_pitch = -center_pitch;
        
        //修正系统偏差（在调试任务二之后得出的参数）
        double yaw_offset = 0.06; 
        double pitch_offset = -0.09;  
        
        target_yaw += yaw_offset;
        target_pitch += pitch_offset;
        
        //发送控制命令
        gimbal.send(true, is_aimed, target_yaw, target_pitch);
        
        //输出Plotter数据
        nlohmann::json plot_data;
        
        //发送给云台的控制命令
        plot_data["gimbal_yaw"] = target_yaw;
        plot_data["gimbal_pitch"] = target_pitch;
        
        //EKF拟合的目标角速度
        plot_data["angular_velocity"] = angular_velocity;
        
        plotter.plot(plot_data);
      } 
      else 
      {
        // 未收敛时保持云台指向EKF预测的中心，但不允许开火
        auto ekf_state = target->ekf_x();
        Eigen::Vector3d center_pos;
        center_pos[0] = ekf_state[0];  
        center_pos[1] = ekf_state[2];  
        center_pos[2] = ekf_state[4];  
        
        auto ypd = tools::xyz2ypd(center_pos);
        double target_yaw = ypd[0];
        double target_pitch = ypd[1];
        
        //反转pitch方向
        target_pitch = -target_pitch;
        
        //修正系统偏差（与任务二保持一致）
        double yaw_offset = 0.06;   
        double pitch_offset = -0.09; 
        
        target_yaw += yaw_offset;
        target_pitch += pitch_offset;
        
        gimbal.send(true, false, target_yaw, target_pitch);
      }
    } else {
      // 没有检测到目标
      if (target_found) {
        target->predict(t);
        gimbal.send(false, false, 0, 0);
      } else {
        gimbal.send(false, false, 0, 0);
      }
    }

    
    cv::waitKey(1);

    // Your code end
  }

  return 0;
}
