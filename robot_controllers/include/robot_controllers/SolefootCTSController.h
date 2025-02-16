#ifndef _LIMX_SOLEFOOT_CTS_CONTROLLER_H_
#define _LIMX_SOLEFOOT_CTS_CONTROLLER_H_

#include "robot_controllers/PointfootController.h"
#include "realtime_tools/realtime_buffer.h"
#include "ros/ros.h"

namespace robot_controller {

class SolefootCTSController : public PointfootController {
  using tensor_element_t = float;
  using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

public:
  SolefootCTSController() = default;
  ~SolefootCTSController() override = default;

  void starting(const ros::Time& time) override;

protected:
  bool loadModel() override;
  bool loadRLCfg() override;
  void computeActions() override;
  void computeObservation() override;
  void handleWalkMode() override;
  void computeLatent();

private:
  // ONNX model paths
  std::string policyPath_;
  std::string encoderPath_;

  // ONNX runtime environment and sessions
  std::shared_ptr<Ort::Env> onnxEnvPtr_;

  // Policy model
  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::vector<const char*> policyInputNames_;
  std::vector<const char*> policyOutputNames_;
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;

  // Encoder model
  std::unique_ptr<Ort::Session> encoderSessionPtr_;
  std::vector<const char*> encoderInputNames_;
  std::vector<const char*> encoderOutputNames_;
  std::vector<std::vector<int64_t>> encoderInputShapes_;
  std::vector<std::vector<int64_t>> encoderOutputShapes_;

  // State dimensions
  int actionsSize_;
  int observationsSize_;
  int obsHistoryLen_;
  int latentDim_;

  // State vectors
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> observations_;
  std::vector<tensor_element_t> latent_;
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> observationsHistoryBuffer_;
  vector_t lastActions_;
  float imu_orientation_offset[3];  // IMU orientation offset

  // Ankle joint parameters
  float ankleJointStiffness_;
  float ankleJointDamping_;

  bool isFirstRecObs_{true};
  double gaitIndex_{0};
  ros::Time timeStamp;
};

} // namespace robot_controller

#endif // _LIMX_SOLEFOOT_CTS_CONTROLLER_H_ 