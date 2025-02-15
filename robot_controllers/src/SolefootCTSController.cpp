#include "robot_controllers/SolefootCTSController.h"
#include <pluginlib/class_list_macros.hpp>

namespace robot_controller {

void SolefootCTSController::starting(const ros::Time& time) {
  for (size_t i = 0; i < hybridJointHandles_.size(); i++) {
    ROS_INFO_STREAM("starting hybridJointHandle: " << hybridJointHandles_[i].getPosition());
    defaultJointAngles_[i] = hybridJointHandles_[i].getPosition();
  }

  standPercent_ += 1 / (standDuration_ * loopFrequency_);
  loopCount_ = 0;
  mode_ = Mode::STAND;
}

void SolefootCTSController::handleWalkMode() {
  if (robotCfg_.controlCfg.decimation == 0) {
    ROS_ERROR("Error robotCfg_.controlCfg.decimation");
    return;
  }

  if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
    computeObservation();
    computeLatent();
    computeActions();

    // Limit action range
    scalar_t actionMin = -robotCfg_.rlCfg.clipActions;
    scalar_t actionMax = robotCfg_.rlCfg.clipActions;
    std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                   [actionMin, actionMax](scalar_t x) { return std::max(actionMin, std::min(actionMax, x)); });
  }

  // Set action
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); i++) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  for (size_t i = 0; i < hybridJointHandles_.size(); i++) {
    if ((i + 1) % 4 != 0) {  // Not ankle joint
      scalar_t actionMin =
          jointPos(i) - initJointAngles_(i, 0) +
          (robotCfg_.controlCfg.damping * jointVel(i) - robotCfg_.controlCfg.user_torque_limit) / robotCfg_.controlCfg.stiffness;
      scalar_t actionMax =
          jointPos(i) - initJointAngles_(i, 0) +
          (robotCfg_.controlCfg.damping * jointVel(i) + robotCfg_.controlCfg.user_torque_limit) / robotCfg_.controlCfg.stiffness;
      actions_[i] = std::max(actionMin / robotCfg_.controlCfg.action_scale_pos,
                             std::min(actionMax / robotCfg_.controlCfg.action_scale_pos, (scalar_t)actions_[i]));
      scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.action_scale_pos + initJointAngles_(i, 0);
      hybridJointHandles_[i].setCommand(pos_des, 0, robotCfg_.controlCfg.stiffness, robotCfg_.controlCfg.damping, 0, 2);
    } else {  // Ankle joint
      scalar_t actionMin = (jointVel(i) - robotCfg_.controlCfg.user_torque_limit / ankleJointDamping_);
      scalar_t actionMax = (jointVel(i) + robotCfg_.controlCfg.user_torque_limit / ankleJointDamping_);
      actions_[i] = std::max(actionMin / ankleJointDamping_, std::min(actionMax / ankleJointDamping_, (scalar_t)actions_[i]));
      scalar_t velocity_des = actions_[i] * ankleJointDamping_;
      hybridJointHandles_[i].setCommand(0, velocity_des, ankleJointStiffness_, ankleJointDamping_, 0, 0);
    }
    lastActions_(i, 0) = actions_[i];
  }
}

void SolefootCTSController::computeActions() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputTensors;

  // Add observations
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observations_.data(), observations_.size(),
                                                                    policyInputShapes_[0].data(), policyInputShapes_[0].size()));

  // Add latent
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, latent_.data(), latent_.size(),
                                                                    policyInputShapes_[1].data(), policyInputShapes_[1].size()));

  // Run policy inference
  Ort::RunOptions runOptions;
  auto outputTensors = policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputTensors.data(), 2, policyOutputNames_.data(), 1);

  // Extract actions
  auto* outputData = outputTensors[0].GetTensorMutableData<tensor_element_t>();
  actions_.resize(actionsSize_);
  std::memcpy(actions_.data(), outputData, actions_.size() * sizeof(tensor_element_t));
}

void SolefootCTSController::computeLatent() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputTensors;

  // Add observation history
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observationsHistoryBuffer_.data(),
                                                                    observationsHistoryBuffer_.size(), encoderInputShapes_[0].data(),
                                                                    encoderInputShapes_[0].size()));

  // Run encoder inference
  Ort::RunOptions runOptions;
  auto outputTensors =
      encoderSessionPtr_->Run(runOptions, encoderInputNames_.data(), inputTensors.data(), 1, encoderOutputNames_.data(), 1);

  // Extract latent
  auto* outputData = outputTensors[0].GetTensorMutableData<tensor_element_t>();
  latent_.resize(latentDim_);
  std::memcpy(latent_.data(), outputData, latent_.size() * sizeof(tensor_element_t));
}

void SolefootCTSController::computeObservation() {
  // Get IMU orientation
  Eigen::Quaterniond q_wi;
  for (size_t i = 0; i < 4; ++i) {
    q_wi.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }

  // Convert quaternion to ZYX Euler angles and calculate inverse rotation matrix
  vector3_t zyx = quatToZyx(q_wi);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  // Define gravity vector and project it to the body frame
  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  // Get base angular velocity
  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);
  vector3_t _zyx(imu_orientation_offset[0], imu_orientation_offset[1], imu_orientation_offset[2]);
  matrix_t rot = getRotationMatrixFromZyxEulerAngles(_zyx);
  baseAngVel = rot * baseAngVel;
  projectedGravity = rot * projectedGravity;

  // Get joint states
  vector_t jointPos(hybridJointHandles_.size());
  vector_t jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  // Get last actions
  vector_t actions(lastActions_);

  // Construct observation vector
  vector_t obs(observationsSize_);
  obs << baseAngVel, projectedGravity, commands_, (jointPos - initJointAngles_), jointVel, actions;

  // Update observations
  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
  }

  // Update observation history buffer
  if (isFirstRecObs_) {
    observationsHistoryBuffer_.resize(obsHistoryLen_ * observationsSize_);
    for (size_t i = 0; i < obsHistoryLen_; i++) {
      std::copy(observations_.begin(), observations_.end(), observationsHistoryBuffer_.data() + i * observationsSize_);
    }
    isFirstRecObs_ = false;
  } else {
    observationsHistoryBuffer_.head(observationsHistoryBuffer_.size() - observationsSize_) =
        observationsHistoryBuffer_.tail(observationsHistoryBuffer_.size() - observationsSize_);
    Eigen::Map<const Eigen::VectorXf> obsVec(observations_.data(), observations_.size());
    observationsHistoryBuffer_.tail(observationsSize_) = obsVec;
  }
}

bool SolefootCTSController::loadModel() {
  // Get model paths
  if (!nh_.getParam("/policyFile", policyPath_) || !nh_.getParam("/encoderFile", encoderPath_)) {
    ROS_ERROR("Failed to retrieve model paths from parameter server!");
    return false;
  }

  // Create ONNX environment
  onnxEnvPtr_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SolefootCTSController"));

  // Create session options
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  sessionOptions.SetInterOpNumThreads(1);

  Ort::AllocatorWithDefaultOptions allocator;

  // Load policy model
  ROS_INFO("Loading policy from: %s", policyPath_.c_str());
  policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, policyPath_.c_str(), sessionOptions);

  // Get policy model info
  for (size_t i = 0; i < policySessionPtr_->GetInputCount(); i++) {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < policySessionPtr_->GetOutputCount(); i++) {
    policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
    policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  // Load encoder model
  ROS_INFO("Loading encoder from: %s", encoderPath_.c_str());
  encoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, encoderPath_.c_str(), sessionOptions);

  // Get encoder model info
  for (size_t i = 0; i < encoderSessionPtr_->GetInputCount(); i++) {
    encoderInputNames_.push_back(encoderSessionPtr_->GetInputName(i, allocator));
    encoderInputShapes_.push_back(encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < encoderSessionPtr_->GetOutputCount(); i++) {
    encoderOutputNames_.push_back(encoderSessionPtr_->GetOutputName(i, allocator));
    encoderOutputShapes_.push_back(encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  ROS_INFO("Successfully loaded ONNX models!");
  return true;
}

bool SolefootCTSController::loadRLCfg() {
  try {
    // Load joint configuration
    if (!nh_.getParam("/SolefootCTSCfg/joint_names", jointNames_)) {
      ROS_ERROR("Failed to load joint names");
      return false;
    }

    // Load initial joint angles
    auto& initState = robotCfg_.initState;
    for (const auto& joint : jointNames_) {
      if (!nh_.getParam("/SolefootCTSCfg/init_state/default_joint_angle/" + joint, initState[joint])) {
        ROS_ERROR_STREAM("Failed to load initial angle for joint: " << joint);
        return false;
      }
    }

    // Load control parameters
    auto& controlCfg = robotCfg_.controlCfg;
    if (!nh_.getParam("/SolefootCTSCfg/control/stiffness", controlCfg.stiffness) ||
        !nh_.getParam("/SolefootCTSCfg/control/damping", controlCfg.damping) ||
        !nh_.getParam("/SolefootCTSCfg/control/ankle_joint_stiffness", ankleJointStiffness_) ||
        !nh_.getParam("/SolefootCTSCfg/control/ankle_joint_damping", ankleJointDamping_) ||
        !nh_.getParam("/SolefootCTSCfg/control/action_scale_pos", controlCfg.action_scale_pos) ||
        !nh_.getParam("/SolefootCTSCfg/control/decimation", controlCfg.decimation) ||
        !nh_.getParam("/SolefootCTSCfg/control/user_torque_limit", controlCfg.user_torque_limit)) {
      ROS_ERROR("Failed to load control parameters");
      return false;
    }

    // Load state dimensions
    if (!nh_.getParam("/SolefootCTSCfg/size/actions_size", actionsSize_) ||
        !nh_.getParam("/SolefootCTSCfg/size/observations_size", observationsSize_) ||
        !nh_.getParam("/SolefootCTSCfg/size/obs_history_length", obsHistoryLen_) ||
        !nh_.getParam("/SolefootCTSCfg/size/latent_dim", latentDim_)) {
      ROS_ERROR("Failed to load state dimensions");
      return false;
    }

    if (!nh_.getParam("/SolefootCTSCfg/imu_orientation_offset/yaw", imu_orientation_offset[0]) ||
        !nh_.getParam("/SolefootCTSCfg/imu_orientation_offset/pitch", imu_orientation_offset[1]) ||
        !nh_.getParam("/SolefootCTSCfg/imu_orientation_offset/roll", imu_orientation_offset[2])) {
      ROS_ERROR("Failed to load imu orientation offset");
      return false;
    }

    // Initialize state vectors
    actions_.resize(actionsSize_);
    observations_.resize(observationsSize_);
    lastActions_.resize(actionsSize_);
    latent_.resize(latentDim_);

    return true;
  } catch (const std::exception& e) {
    ROS_ERROR("Error in loading RL config: %s", e.what());
    return false;
  }
}

}  // namespace robot_controller

PLUGINLIB_EXPORT_CLASS(robot_controller::SolefootCTSController, controller_interface::ControllerBase)