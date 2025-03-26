#include "robot_controllers/PointfootCTSVisionController.h"

#include <angles/angles.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/Marker.h>

#include <iterator>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace robot_controller {

void PointfootCTSVisionController::starting(const ros::Time& time) {
  PointfootController::starting(time);
  // Start depth backbone computation thread
  computeDepthBackboneThread_ = std::thread(&PointfootCTSVisionController::computeDepthBackbone, this);
}

void PointfootCTSVisionController::stopping(const ros::Time& time) {
  exitSignal_.set_value();
  if (computeDepthBackboneThread_.joinable()) {
    computeDepthBackboneThread_.join();
  }
}

void PointfootCTSVisionController::handleWalkMode() {
  // Check decimation parameter
  if (robotCfg_.controlCfg.decimation == 0) {
    ROS_ERROR("Error robotCfg_.controlCfg.decimation");
    return;
  }

  // Compute observation & actions at specified decimation rate
  if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
    computeObservation();

    // Write data to sub-thread buffer
    main2SubBuffer_.writeFromNonRT(Main2SubParams(normalizedObs_, observationsHistoryBuffer_));

    // Read processed data from sub-thread
    auto subData = sub2MainBuffer_.readFromRT();
    if (subData) {
      hmLatent_ = subData->hmLatent;
      predictedHeightMap_ = subData->predictedHeightMap;
    }

    // Skip if no height map data available
    if (predictedHeightMap_.empty()) {
      ROS_INFO("[CTS Vision] No height map data available");
      return;
    }

    // Compute EP estimate、
    computeEpEstimate();

    // Compute final actions
    computeActions();
  }

  // Apply actions to joints
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); i++) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  for (int i = 0; i < hybridJointHandles_.size(); i++) {
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

    lastActions_(i, 0) = actions_[i];
  }
}

void PointfootCTSVisionController::computeActions() {
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::vector<Ort::Value> inputTensors;

  // Add normalized observations
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, normalizedObs_.data(), normalizedObs_.size(),
                                                                    policyInputShapes_[0].data(), policyInputShapes_[0].size()));

  // Add EP latent
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, epLatent_.data(), epLatent_.size(),
                                                                    policyInputShapes_[1].data(), policyInputShapes_[1].size()));

  // Add HM latent
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, hmLatent_.data(), hmLatent_.size(),
                                                                    policyInputShapes_[2].data(), policyInputShapes_[2].size()));

  // Run policy inference
  Ort::RunOptions runOptions;
  auto outputTensors = policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputTensors.data(), 3, policyOutputNames_.data(), 1);

  // Extract actions
  auto* outputData = outputTensors[0].GetTensorMutableData<tensor_element_t>();
  actions_.resize(actionsSize_);
  std::memcpy(actions_.data(), outputData, actions_.size() * sizeof(tensor_element_t));

  // Clip actions
  scalar_t actionMin = -robotCfg_.rlCfg.clipActions;
  scalar_t actionMax = robotCfg_.rlCfg.clipActions;
  std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                 [actionMin, actionMax](scalar_t x) { return std::max(actionMin, std::min(actionMax, x)); });

  // End timing and print duration in milliseconds
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  ROS_INFO_THROTTLE(1.0, "[Timing] Policy inference took %.3f ms", duration.count() / 1000.0);
}

void PointfootCTSVisionController::computeDepthBackbone() {
  while (futureObj_.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout) {
    // Get latest data from main thread
    auto mainParams = *main2SubBuffer_.readFromNonRT();
    auto depthImage = *latestDepthImageBuffer_.readFromNonRT();

    if (mainParams.normalizedObs.empty() || mainParams.normalizedObsHistory.size() == 0 || depthImage.empty()) {
      ROS_INFO_THROTTLE(1.0, "[CTS Vision] Waiting for valid data");
      continue;
    }

    // Prepare input tensors
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;

    // Add proprioceptive observations history
    inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, mainParams.normalizedObsHistory.data(), mainParams.normalizedObsHistory.size(), depthBackboneInputShapes_[0].data(),
        depthBackboneInputShapes_[0].size()));

    // Add depth image
    inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, depthImage.data(), depthImage.size(), depthBackboneInputShapes_[1].data(), depthBackboneInputShapes_[1].size()));

    // Add hidden states
    inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, hiddenStates_.data(), hiddenStates_.size(), depthBackboneInputShapes_[2].data(), depthBackboneInputShapes_[2].size()));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Run depth backbone inference
    Ort::RunOptions runOptions;
    auto outputTensors = depthBackboneSessionPtr_->Run(runOptions, depthBackboneInputNames_.data(), inputTensors.data(), 3,
                                                       depthBackboneOutputNames_.data(), 3);

    // End timing and print duration in milliseconds
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    ROS_INFO_THROTTLE(1.0, "[Timing] Depth backbone inference took %.3f ms", duration.count() / 1000.0);

    // Extract HM latent and predicted height map
    hmLatent_.resize(hmLatentDim_);
    predictedHeightMap_.resize(outputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount());

    std::memcpy(hmLatent_.data(), outputTensors[0].GetTensorData<tensor_element_t>(), hmLatent_.size() * sizeof(tensor_element_t));
    std::memcpy(predictedHeightMap_.data(), outputTensors[1].GetTensorData<tensor_element_t>(),
                predictedHeightMap_.size() * sizeof(tensor_element_t));

    // Update hidden states
    hiddenStates_.resize(hiddenStatesSize_);
    std::memcpy(hiddenStates_.data(), outputTensors[2].GetTensorData<tensor_element_t>(), hiddenStates_.size() * sizeof(tensor_element_t));

    // Write results back to main thread
    sub2MainBuffer_.writeFromNonRT(Sub2MainParams(hmLatent_, predictedHeightMap_));

    // Visualize height map if needed
    visualizeHeightMap(predictedHeightMap_);
  }
}

void PointfootCTSVisionController::computeEpEstimate() {
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  // Prepare input tensor with observation history
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::vector<Ort::Value> inputTensors;

  // Create tensor from observation history buffer
  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observationsHistoryBuffer_.data(),
                                                                    observationsHistoryBuffer_.size(), epEstimatorInputShapes_[0].data(),
                                                                    epEstimatorInputShapes_[0].size()));

  // Run EP estimator inference
  Ort::RunOptions runOptions;
  auto outputTensors =
      epEstimatorSessionPtr_->Run(runOptions, epEstimatorInputNames_.data(), inputTensors.data(), 1, epEstimatorOutputNames_.data(), 1);

  // Extract EP latent
  auto* outputData = outputTensors[0].GetTensorMutableData<tensor_element_t>();
  epLatent_.resize(epLatentDim_);
  std::memcpy(epLatent_.data(), outputData, epLatent_.size() * sizeof(tensor_element_t));

  // End timing and print duration in milliseconds
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  ROS_INFO_THROTTLE(1.0, "[Timing] EP estimator inference took %.3f ms", duration.count() / 1000.0);
}

void PointfootCTSVisionController::computeObservation() {
  // Get IMU orientation
  Eigen::Quaterniond q_wi;
  for (size_t i = 0; i < 4; ++i) {
    q_wi.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }

  // Convert quaternion to ZYX Euler angles and calculate inverse rotation matrix
  vector3_t zyx = quatToZyx(q_wi);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  // Project gravity vector to body frame
  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  // Get and transform base angular velocity
  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);
  vector3_t _zyx(imuOrientationOffset_[0], imuOrientationOffset_[1], imuOrientationOffset_[2]);
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

  // Calculate gait phase
  vector_t gait(3);
  gait << 2.0, 0.5, 0.5;  // trot gait parameters
  gaitIndex_ += 0.02 * gait(0);
  if (gaitIndex_ > 1.0) {
    gaitIndex_ = 0.0;
  }
  vector_t gait_clock(2);
  gait_clock << sin(gaitIndex_ * 2 * M_PI), cos(gaitIndex_ * 2 * M_PI);

  // Get last actions
  vector_t actions(lastActions_);

  // Construct observation vector
  vector_t obs(observationsSize_);
  // obs << baseAngVel, projectedGravity, commands_, (jointPos - initJointAngles_), jointVel, actions, gait_clock, gait;
  obs << baseAngVel, projectedGravity, commands_, (jointPos - initJointAngles_), jointVel, actions;

  // Update raw observations
  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
  }

  // Compute normalized observations
  computeNormalizedObs(obs);

  // Update observation history buffer with normalized observations
  if (isFirstRecObs_) {
    observationsHistoryBuffer_.resize(obsHistoryLen_ * observationsSize_);
    for (size_t i = 0; i < obsHistoryLen_; i++) {
      std::copy(normalizedObs_.begin(), normalizedObs_.end(), observationsHistoryBuffer_.data() + i * observationsSize_);
    }
    isFirstRecObs_ = false;
  } else {
    observationsHistoryBuffer_.head(observationsHistoryBuffer_.size() - observationsSize_) =
        observationsHistoryBuffer_.tail(observationsHistoryBuffer_.size() - observationsSize_);
    Eigen::Map<const Eigen::VectorXf> normalizedObsVec(normalizedObs_.data(), normalizedObs_.size());
    observationsHistoryBuffer_.tail(observationsSize_) = normalizedObsVec;
  }
}

void PointfootCTSVisionController::computeNormalizedObs(const vector_t& obs) {
  // Prepare input tensor
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::vector<Ort::Value> inputTensors;
  std::vector<tensor_element_t> obsData(obs.size());
  for (size_t i = 0; i < obs.size(); i++) {
    obsData[i] = static_cast<tensor_element_t>(obs(i));
  }

  inputTensors.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, obsData.data(), obsData.size(),
                                                                    normalizerInputShapes_[0].data(), normalizerInputShapes_[0].size()));

  // Run normalizer inference
  Ort::RunOptions runOptions;
  auto outputTensors =
      normalizerSessionPtr_->Run(runOptions, normalizerInputNames_.data(), inputTensors.data(), 1, normalizerOutputNames_.data(), 1);

  // Extract normalized observations
  auto* outputData = outputTensors[0].GetTensorMutableData<tensor_element_t>();
  normalizedObs_.resize(obs.size());
  std::memcpy(normalizedObs_.data(), outputData, normalizedObs_.size() * sizeof(tensor_element_t));

  // Clip normalized observations
  scalar_t obsMin = -robotCfg_.rlCfg.clipObs;
  scalar_t obsMax = robotCfg_.rlCfg.clipObs;
  std::transform(normalizedObs_.begin(), normalizedObs_.end(), normalizedObs_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

bool PointfootCTSVisionController::loadModel() {
  ROS_INFO("[CTS Vision] Loading models");
  try {
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;

    // Create ONNX Runtime environment
    onnxEnvPtr_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "PointfootCTSVisionController"));

    // Get model paths from parameter server
    if (!nh_.getParam("/normalizerFile", normalizerPath_) || !nh_.getParam("/policyFile", policyPath_) ||
        !nh_.getParam("/depthBackboneFile", depthBackbonePath_) || !nh_.getParam("/epEstimatorFile", epEstimatorPath_)) {
      ROS_ERROR("Failed to get model paths from parameter server");
      return false;
    }

    // Load normalizer model
    ROS_INFO_STREAM("Loading normalizer model from: " << normalizerPath_);
    normalizerSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, normalizerPath_.c_str(), sessionOptions);

    // Get normalizer model info
    for (size_t i = 0; i < normalizerSessionPtr_->GetInputCount(); i++) {
      normalizerInputNames_.push_back(normalizerSessionPtr_->GetInputName(i, allocator));
      normalizerInputShapes_.push_back(normalizerSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < normalizerSessionPtr_->GetOutputCount(); i++) {
      normalizerOutputNames_.push_back(normalizerSessionPtr_->GetOutputName(i, allocator));
      normalizerOutputShapes_.push_back(normalizerSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    // Load policy model
    ROS_INFO_STREAM("Loading policy model from: " << policyPath_);
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

    // Load depth backbone model
    ROS_INFO_STREAM("Loading depth backbone model from: " << depthBackbonePath_);
    depthBackboneSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, depthBackbonePath_.c_str(), sessionOptions);

    // Get depth backbone model info
    for (size_t i = 0; i < depthBackboneSessionPtr_->GetInputCount(); i++) {
      depthBackboneInputNames_.push_back(depthBackboneSessionPtr_->GetInputName(i, allocator));
      depthBackboneInputShapes_.push_back(depthBackboneSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < depthBackboneSessionPtr_->GetOutputCount(); i++) {
      depthBackboneOutputNames_.push_back(depthBackboneSessionPtr_->GetOutputName(i, allocator));
      depthBackboneOutputShapes_.push_back(depthBackboneSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    // Load EP estimator model
    ROS_INFO_STREAM("Loading EP estimator model from: " << epEstimatorPath_);
    epEstimatorSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, epEstimatorPath_.c_str(), sessionOptions);

    // Get EP estimator model info
    for (size_t i = 0; i < epEstimatorSessionPtr_->GetInputCount(); i++) {
      epEstimatorInputNames_.push_back(epEstimatorSessionPtr_->GetInputName(i, allocator));
      epEstimatorInputShapes_.push_back(epEstimatorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < epEstimatorSessionPtr_->GetOutputCount(); i++) {
      epEstimatorOutputNames_.push_back(epEstimatorSessionPtr_->GetOutputName(i, allocator));
      epEstimatorOutputShapes_.push_back(epEstimatorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    ROS_INFO("Successfully loaded all models");
    return true;

  } catch (const Ort::Exception& e) {
    ROS_ERROR_STREAM("ONNX Runtime error: " << e.what());
    return false;
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("Error loading models: " << e.what());
    return false;
  }
}

bool PointfootCTSVisionController::loadRLCfg() {
  try {
    // Load joint configuration
    if (!nh_.getParam("/PointfootCTSVisionCfg/joint_names", jointNames_)) {
      ROS_ERROR("Failed to load joint names");
      return false;
    }

    // Load initial joint angles
    auto& initState = robotCfg_.initState;
    for (const auto& joint : jointNames_) {
      if (!nh_.getParam("/PointfootCTSVisionCfg/init_state/default_joint_angle/" + joint, initState[joint])) {
        ROS_ERROR_STREAM("Failed to load initial angle for joint: " << joint);
        return false;
      }
    }

    // Load normalization parameters
    auto& rlCfg = robotCfg_.rlCfg;
    if (!nh_.getParam("/PointfootCTSVisionCfg/normalization/clip_scales/clip_actions", rlCfg.clipActions) ||
        !nh_.getParam("/PointfootCTSVisionCfg/normalization/clip_scales/clip_observations", rlCfg.clipObs)) {
      ROS_ERROR("Failed to load clip scales parameters");
      return false;
    }

    // Load control parameters
    auto& controlCfg = robotCfg_.controlCfg;
    if (!nh_.getParam("/PointfootCTSVisionCfg/control/stiffness", controlCfg.stiffness) ||
        !nh_.getParam("/PointfootCTSVisionCfg/control/damping", controlCfg.damping) ||
        !nh_.getParam("/PointfootCTSVisionCfg/control/action_scale_pos", controlCfg.action_scale_pos) ||
        !nh_.getParam("/PointfootCTSVisionCfg/control/decimation", controlCfg.decimation) ||
        !nh_.getParam("/PointfootCTSVisionCfg/control/user_torque_limit", controlCfg.user_torque_limit)) {
      ROS_ERROR("Failed to load control parameters");
      return false;
    }

    // Load state dimensions
    if (!nh_.getParam("/PointfootCTSVisionCfg/size/actions_size", actionsSize_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/size/commands_size", commandsSize_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/size/observations_size", observationsSize_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/size/obs_history_length", obsHistoryLen_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/size/ep_latent_dim", epLatentDim_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/size/hm_latent_dim", hmLatentDim_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/size/hidden_state_size", hiddenStatesSize_)) {
      ROS_ERROR("Failed to load state dimensions");
      return false;
    }

    // Load depth image parameters
    if (!nh_.getParam("/PointfootCTSVisionCfg/depth_image/original", depthOriginalShape_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/depth_image/resized", depthResizedShape_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/depth_image/near_clip", nearClip_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/depth_image/far_clip", farClip_) ||
        !nh_.getParam("/PointfootCTSVisionCfg/depth_image/buffer_len", depthBufferLen_)) {
      ROS_ERROR("Failed to load depth image parameters");
      return false;
    }

    // Initialize state vectors
    actions_.resize(actionsSize_);
    observations_.resize(observationsSize_);
    lastActions_.resize(actionsSize_);
    hiddenStates_.resize(hiddenStatesSize_);
    epLatent_.resize(epLatentDim_);
    hmLatent_.resize(hmLatentDim_);

    numPixels_ = depthResizedShape_[0] * depthResizedShape_[1];

    // Initialize ROS communication
    std::string depthImageTopic;
    if (!nh_.getParam("/depthImageTopicName", depthImageTopic)) {
      ROS_ERROR("Failed to get depth image topic name");
      return false;
    }

    resizedDepthImagePub_ = nh_.advertise<sensor_msgs::Image>("/d435F/aligned_depth_to_color/image_resized", 1, true);
    heightMapPub_ = nh_.advertise<visualization_msgs::Marker>("/height_map", 1, true);
    depthImageSub_ = nh_.subscribe(depthImageTopic, 1, &PointfootCTSVisionController::depthImageCallback, this);

    ROS_INFO("Successfully loaded all configuration parameters");
    return true;

  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("Error loading configuration: " << e.what());
    return false;
  }
}

void PointfootCTSVisionController::depthImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  if (msg->encoding != "16UC1") {
    ROS_ERROR_THROTTLE(1.0, "Depth image must be 16UC1 encoded");
    return;
  }

  // Covert origin depth image to vector of floats
  uint32_t imageWidth = msg->width;
  uint32_t imageHeight = msg->height;
  std::vector<float> originalData;
  originalData.reserve(msg->width * msg->height);

  // Convert depth image to meters
  for (size_t i = 0; i < imageWidth * imageHeight; i++) {
    uint16_t pixelValue = (msg->data[i * 2 + 1] << 8) | msg->data[i * 2];
    float distance = static_cast<float>(pixelValue) / 1000.0f;
    distance = std::min(std::max(distance, nearClip_), farClip_);
    originalData.push_back(distance);
  }

  // Create a vector for resize image
  std::vector<float> imageData;
  imageData.reserve(depthResizedShape_[0] * depthResizedShape_[1]);

  // Resize depth image if needed
  if (msg->width != depthResizedShape_[1] || msg->height != depthResizedShape_[0]) {
    cv::Mat depth_mat(msg->height, msg->width, CV_32FC1, originalData.data());
    cv::Mat resized_mat;
    cv::resize(depth_mat, resized_mat, cv::Size(depthResizedShape_[1], depthResizedShape_[0]), 0, 0, cv::INTER_AREA);
    imageData.assign((float*)resized_mat.datastart, (float*)resized_mat.dataend);
  } else {
    imageData = originalData;
  }

  // Crop 20 pixels from the width dimension
  std::vector<float> cropimageData;
  const size_t crop_size = depthResizedShape_[0] * (depthResizedShape_[1] - 20);
  cropimageData.resize(crop_size);

  cv::Mat depth_mat(depthResizedShape_[0], depthResizedShape_[1], CV_32FC1, imageData.data());
  cv::Mat cropped_mat = depth_mat(cv::Range::all(), cv::Range(20, depthResizedShape_[1])).clone();

  std::memcpy(cropimageData.data(), cropped_mat.data, crop_size * sizeof(float));

  // Update depth image buffer
  latestDepthImageBuffer_.writeFromNonRT(cropimageData);

  // Publish resized depth image for visualization
  sensor_msgs::Image resized_msg;
  resized_msg.header = msg->header;
  resized_msg.encoding = "32FC1";
  resized_msg.height = depthResizedShape_[0];
  resized_msg.width = depthResizedShape_[1] - 20;
  resized_msg.is_bigendian = false;
  resized_msg.step = resized_msg.width * sizeof(float);

  const size_t data_size = resized_msg.height * resized_msg.width * sizeof(float);
  resized_msg.data.resize(data_size);

  std::memcpy(resized_msg.data.data(), cropimageData.data(), data_size);

  resizedDepthImagePub_.publish(resized_msg);
}

void PointfootCTSVisionController::visualizeHeightMap(const std::vector<tensor_element_t>& heightMap) {
  // Initialize marker message
  visualization_msgs::Marker points;
  points.header.frame_id = "base_link";
  points.header.stamp = ros::Time::now();
  points.ns = "height_map";
  points.action = visualization_msgs::Marker::ADD;
  points.pose.orientation.w = 1.0;
  points.id = 0;
  points.type = visualization_msgs::Marker::POINTS;

  // Set the scale of the marker points
  points.scale.x = 0.02;  // Point width
  points.scale.y = 0.02;  // Point height

  // Set the color (green with full opacity)
  points.color.g = 1.0f;
  points.color.a = 1.0;

  // Define grid parameters matching the configuration file
  const float resolution = 0.05f;  // Grid resolution in meters
  const float width = 1.0f;        // Total width (x-direction) in meters
  const float height = 1.0f;       // Total height (y-direction) in meters
  const float x_offset = -0.5f;    // Offset in x-direction from base frame

  // Calculate number of points in each dimension
  const int width_points = static_cast<int>(width / resolution) + 1;    // Number of points along width
  const int height_points = static_cast<int>(height / resolution) + 1;  // Number of points along height

  // Verify heightMap size matches expected dimensions
  if (heightMap.size() != width_points * height_points) {
    ROS_ERROR_STREAM("Height map size mismatch. Expected: " << width_points * height_points << ", Got: " << heightMap.size());
    return;
  }

  // Create points from height map
  for (int i = 0; i < height_points; ++i) {
    for (int j = 0; j < width_points; ++j) {
      geometry_msgs::Point p;

      // Calculate point coordinates:
      // x: starts from x_offset and extends by width
      // y: centered around 0, extends from -height/2 to height/2
      // z: taken from height map
      p.x = x_offset + j * resolution;
      p.y = (i * resolution) - (height / 2.0f);
      p.z = -heightMap[i * width_points + j] - 0.5;

      // Add point to marker
      points.points.push_back(p);
    }
  }

  // Publish the marker
  heightMapPub_.publish(points);
}

}  // namespace robot_controller

// Export controller plugin
PLUGINLIB_EXPORT_CLASS(robot_controller::PointfootCTSVisionController, controller_interface::ControllerBase)