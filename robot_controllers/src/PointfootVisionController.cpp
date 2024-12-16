//
// Created by suhaokai on 24-10-17.
//

#include "robot_controllers/PointfootVisionController.h"

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
void PointfootVisionController::starting(const ros::Time& time) {
  PointfootController::starting(time);
  // computePredictedHeightMapThread_ =
  // std::thread(&PointfootVisionController::computeDepthBackBone, this);
  computePredictedHeightMapThread_ = std::thread(&PointfootVisionController::computePredictedHeightMap, this);
}

void PointfootVisionController::stopping(const ros::Time& time) {
  exitSignal_.set_value();
  if (computePredictedHeightMapThread_.joinable()) {
    computePredictedHeightMapThread_.join();
  }
}

// Handle walking mode
void PointfootVisionController::handleWalkMode() {
  // Compute observation & actions
  if (robotCfg_.controlCfg.decimation == 0) {
    ROS_ERROR("Error robotCfg_.controlCfg.decimation");
    return;
  }
  if (loopCount_ % robotCfg_.controlCfg.decimation == 0) {
    computeObservation();

    computeLinVelEstimate();

    computeEnvParamsEstimate();

    main2SubBuffer_.writeFromNonRT(Main2SubParams(linVelLatent_, observations_, scaled_commands_));
    predictedHeightMap_ = sub2MainBuffer_.readFromRT()->predictedHeightMap;

    if (predictedHeightMap_.empty()) {
      return;
    }

    computeHeightMapEncoder();

    computeBeliefEncoder();

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

void PointfootVisionController::computeActions() {
  // combine observation and depth latent
  std::vector<tensor_element_t> combineData;
  for (size_t i = 0; i < linVelLatent_.size(); i++) {
    combineData.push_back(linVelLatent_[i]);
  }
  for (size_t i = 0; i < beliefLatents_.size(); i++) {
    combineData.push_back(beliefLatents_[i]);
  }
  for (size_t i = 0; i < envParamsLatent_.size(); i++) {
    combineData.push_back(envParamsLatent_[i]);
  }
  for (size_t i = 0; i < scaled_commands_.size(); i++) {
    combineData.push_back(scaled_commands_[i]);
  }
  for (size_t i = 0; i < observations_.size(); i++) {
    combineData.push_back(observations_[i]);
  }

  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combineData.data(), combineData.size(),
                                                                   policyInputShapes_[0].data(), policyInputShapes_[0].size()));
  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues =
      policySessionPtr_->Run(runOptions, policyInputNames_.data(), inputValues.data(), 1, policyOutputNames_.data(), 1);

  for (size_t i = 0; i < actionsSize_; i++) {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void PointfootVisionController::computeObservation() {
  // Get IMU orientation
  Eigen::Quaterniond q_wi;
  for (size_t i = 0; i < 4; ++i) {
    q_wi.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }
  // Convert quaternion to ZYX Euler angles and calculate inverse rotation
  // matrix
  vector3_t zyx = quatToZyx(q_wi);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  // Define gravity vector and project it to the body frame
  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  // Get base angular velocity and apply orientation offset
  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);
  vector3_t _zyx(imuOrientationOffset_[0], imuOrientationOffset_[1], imuOrientationOffset_[2]);
  matrix_t rot = getRotationMatrixFromZyxEulerAngles(_zyx);
  baseAngVel = rot * baseAngVel;
  projectedGravity = rot * projectedGravity;

  // Get initial state of joints
  auto& initState = robotCfg_.initState;
  vector_t jointPos(initState.size());
  vector_t jointVel(initState.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  vector_t gait(4);
  gait << 2.0, 0.5, 0.5, 0.1;  // trot
  gaitIndex_ += 0.02 * gait(0);
  if (gaitIndex_ > 1.0) {
    gaitIndex_ = 0.0;
  }
  vector_t gait_clock(2);
  gait_clock << sin(gaitIndex_ * 2 * M_PI), cos(gaitIndex_ * 2 * M_PI);

  vector_t actions(lastActions_);

  // Define command scaler and observation vector
  matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(robotCfg_.rlCfg.obsScales.linVel, robotCfg_.rlCfg.obsScales.linVel,
                                                              robotCfg_.rlCfg.obsScales.angVel);

  vector_t obs(observationsSize_);
  vector3_t scaled_commands = commandScaler * commands_;
  // Populate observation vector
  // clang-format off
  obs << baseAngVel * robotCfg_.rlCfg.obsScales.angVel,
      projectedGravity,
      (jointPos - initJointAngles_) * robotCfg_.rlCfg.obsScales.dofPos,
      jointVel * robotCfg_.rlCfg.obsScales.dofVel,
      actions,
      gait_clock,
      gait;
  // clang-format on

  // Update observation, scaled commands, and proprioceptive history vector
  if (isFirstRecObs_) {
    observationsHistoryBuffer_.resize(obsHistoryLen_ * observationsSize_);
    for (size_t i = 0; i < obsHistoryLen_; i++) {
      std::copy(obs.data(), obs.data() + observationsSize_, observationsHistoryBuffer_.data() + i * observationsSize_);
    }
    isFirstRecObs_ = false;
  }
  observationsHistoryBuffer_.head(observationsHistoryBuffer_.size() - observationsSize_) =
      observationsHistoryBuffer_.tail(observationsHistoryBuffer_.size() - observationsSize_);
  observationsHistoryBuffer_.tail(observationsSize_) = obs.cast<tensor_element_t>();

  for (size_t i = 0; i < obs.size(); i++) {
    observations_[i] = static_cast<tensor_element_t>(obs(i));
  }
  for (size_t i = 0; i < scaled_commands_.size(); i++) {
    scaled_commands_[i] = static_cast<tensor_element_t>(scaled_commands(i));
  }

  // Limit observation range
  scalar_t obsMin = -robotCfg_.rlCfg.clipObs;
  scalar_t obsMax = robotCfg_.rlCfg.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(),
                 [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

void PointfootVisionController::computeLinVelEstimate() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observationsHistoryBuffer_.data(),
                                                                   observationsHistoryBuffer_.size(), linVelEstimatorInputShapes_[0].data(),
                                                                   linVelEstimatorInputShapes_[0].size()));

  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = linVelEstimatorSessionPtr_->Run(runOptions, linVelEstimatorInputNames_.data(), inputValues.data(),
                                                                         1, linVelEstimatorOutputNames_.data(), 1);

  int outputSize = linVelEstimatorOutputShapes_[0][0];
  linVelLatent_.resize(outputSize);
  for (size_t i = 0; i < outputSize; i++) {
    linVelLatent_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void PointfootVisionController::computeEnvParamsEstimate() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<tensor_element_t> combineData;
  for (size_t i = 0; i < observationsHistoryBuffer_.size(); i++) {
    combineData.push_back(observationsHistoryBuffer_(i));
  }
  for (size_t i = 0; i < scaled_commands_.size(); i++) {
    combineData.push_back(scaled_commands_[i]);
  }

  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combineData.data(), combineData.size(),
                                                                   envParamsEstimatorInputShapes_[0].data(),
                                                                   envParamsEstimatorInputShapes_[0].size()));

  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = envParamsEstimatorSessionPtr_->Run(
      runOptions, envParamsEstimatorInputNames_.data(), inputValues.data(), 1, envParamsEstimatorOutputNames_.data(), 1);

  int outputSize = envParamsEstimatorOutputShapes_[0][0];
  tensor_element_t outputNorm = 0;
  envParamsLatent_.resize(outputSize);
  for (size_t i = 0; i < outputSize; i++) {
    envParamsLatent_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    outputNorm += envParamsLatent_[i] * envParamsLatent_[i];
  }
  if (robotCfg_.rlCfg.encoder_nomalize) {
    for (size_t i = 0; i < outputSize; i++) {
      envParamsLatent_[i] /= std::sqrt(outputNorm);
    }
  }
}

void PointfootVisionController::computePredictedHeightMap() {
  while (futureObj_.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout) {
    Main2SubParams syncParams = *main2SubBuffer_.readFromNonRT();
    std::vector<tensor_element_t> lastImage = *latestDepthImageBuffer_.readFromNonRT();

    if (syncParams.linVelLatent.empty() || syncParams.observation.empty() || lastImage.empty()) continue;

    std::vector<tensor_element_t> robotState;
    for (tensor_element_t i : syncParams.observation) {
      robotState.push_back(i);
    }
    for (tensor_element_t i : syncParams.linVelLatent) {
      robotState.push_back(i);
    }
    for (size_t i = 0; i < syncParams.scaledCommand.size(); i++) {
      robotState.push_back(syncParams.scaledCommand[i]);
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> inputValues;
    inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, lastImage.data(), numPixels_, depthPredictorInputShapes_[0].data(), depthPredictorInputShapes_[0].size()));

    inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, robotState.data(), robotState.size(), depthPredictorInputShapes_[1].data(), depthPredictorInputShapes_[1].size()));

    inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, hiddenStates_.data(), hiddenStatesSize_, depthPredictorInputShapes_[2].data(), depthPredictorInputShapes_[2].size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues = depthPredictorSessionPtr_->Run(runOptions, depthPredictorInputNames_.data(), inputValues.data(),
                                                                          3, depthPredictorOutputNames_.data(), 2);

    int outputSize = depthPredictorOutputShapes_[0][1];
    std::vector<tensor_element_t> predictedHeightMap(outputSize);
    for (size_t i = 0; i < outputSize; i++) {
      predictedHeightMap[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    }

    for (size_t i = 0; i < hiddenStatesSize_; i++) {
      hiddenStates_[i] = *(outputValues[1].GetTensorMutableData<tensor_element_t>() + i);
    }
    sub2MainBuffer_.writeFromNonRT(predictedHeightMap);

    visualizeHeightMap(predictedHeightMap);
  }
}

void PointfootVisionController::computeHeightMapEncoder() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, predictedHeightMap_.data(), predictedHeightMap_.size(),
                                                                   heightMapEncoderInputShapes_[0].data(),
                                                                   heightMapEncoderInputShapes_[0].size()));
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = heightMapEncoderSessionPtr_->Run(runOptions, heightMapEncoderInputNames_.data(),
                                                                          inputValues.data(), 1, heightMapEncoderOutputNames_.data(), 1);
  int outputSize = heightMapEncoderOutputShapes_[0][0];
  heightMapLatent_.resize(outputSize);
  for (size_t i = 0; i < outputSize; i++) {
    heightMapLatent_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void PointfootVisionController::computeBeliefEncoder() {
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  std::vector<tensor_element_t> commandsData;

  for (size_t i = 0; i < scaled_commands_.size(); i++) {
    commandsData.push_back(scaled_commands_[i]);
  }

  // The input of the belief encoder includes proprio_obs, commands, lin_vel,
  // depth_latent, hidden_states
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observations_.data(), observations_.size(),
                                                                   depthBeliefEncoderInputShapes_[0].data(),
                                                                   depthBeliefEncoderInputShapes_[0].size()));

  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, commandsData.data(), commandsData.size(),
                                                                   depthBeliefEncoderInputShapes_[1].data(),
                                                                   depthBeliefEncoderInputShapes_[1].size()));

  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, linVelLatent_.data(), linVelLatent_.size(),
                                                                   depthBeliefEncoderInputShapes_[2].data(),
                                                                   depthBeliefEncoderInputShapes_[2].size()));

  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, heightMapLatent_.data(), heightMapLatent_.size(),
                                                                   depthBeliefEncoderInputShapes_[3].data(),
                                                                   depthBeliefEncoderInputShapes_[3].size()));

  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, beliefHiddenStates_.data(), beliefHiddenStates_.size(),
                                                                   depthBeliefEncoderInputShapes_[4].data(),
                                                                   depthBeliefEncoderInputShapes_[4].size()));

  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = depthBeliefEncoderSessionPtr_->Run(
      runOptions, depthBeliefEncoderInputNames_.data(), inputValues.data(), 5, depthBeliefEncoderOutputNames_.data(), 3);

  int outputSize = depthBeliefEncoderOutputShapes_[2][1];

  beliefLatents_.resize(outputSize);
  for (size_t i = 0; i < outputSize; i++) {
    beliefLatents_[i] = *(outputValues[2].GetTensorMutableData<tensor_element_t>() + i);
  }
  for (size_t i = 0; i < beliefHiddenStatesSize_; i++) {
    beliefHiddenStates_[i] = *(outputValues[1].GetTensorMutableData<tensor_element_t>() + i);
  }
}

bool PointfootVisionController::loadModel() {
  // create env
  onnxEnvPtr_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  Ort::AllocatorWithDefaultOptions allocator;

  ROS_INFO_STREAM("------- load biped vision model -------");
  std::string policyFilePath;
  if (!nh_.getParam("/policyFile", policyFilePath)) {
    ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
    return false;
  }
  policyFilePath_ = policyFilePath;
  ROS_INFO_STREAM("Load biped vision policy model from path : " << policyFilePath);

  ROS_INFO_STREAM("------- load depth predictor model -------");
  std::string depthPredictorPath;
  if (!nh_.getParam("/depthPredictorFile", depthPredictorPath)) {
    ROS_ERROR_STREAM("Get depth predictor path fail from param server, some error occur!");
    return false;
  }
  depthPredictorPath_ = depthPredictorPath;
  ROS_INFO_STREAM("Load depth predictor model from path : " << depthPredictorPath_);

  // depth predictor session
  depthPredictorInputNames_.clear();
  depthPredictorOutputNames_.clear();
  depthPredictorInputShapes_.clear();
  depthPredictorOutputShapes_.clear();
  depthPredictorSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, depthPredictorPath_.c_str(), sessionOptions);
  for (size_t i = 0; i < depthPredictorSessionPtr_->GetInputCount(); i++) {
    depthPredictorInputNames_.push_back(depthPredictorSessionPtr_->GetInputName(i, allocator));
    depthPredictorInputShapes_.push_back(depthPredictorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < depthPredictorSessionPtr_->GetOutputCount(); i++) {
    depthPredictorOutputNames_.push_back(depthPredictorSessionPtr_->GetOutputName(i, allocator));
    depthPredictorOutputShapes_.push_back(depthPredictorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load depth encoder model successfully !!!");

  ROS_INFO_STREAM("------- load height map encoder model -------");
  std::string heightMapEncoderPath;
  if (!nh_.getParam("/heightMapEncoderFile", heightMapEncoderPath)) {
    ROS_ERROR_STREAM(
        "Get height map encoder path fail from param server, some "
        "error occur!");
    return false;
  }
  heightMapEncoderPath_ = heightMapEncoderPath;
  ROS_INFO_STREAM("Load height map encoder model from path : " << heightMapEncoderPath);

  // height map encoder session
  heightMapEncoderInputNames_.clear();
  heightMapEncoderOutputNames_.clear();
  heightMapEncoderInputShapes_.clear();
  heightMapEncoderOutputShapes_.clear();
  heightMapEncoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, heightMapEncoderPath_.c_str(), sessionOptions);
  for (size_t i = 0; i < heightMapEncoderSessionPtr_->GetInputCount(); i++) {
    heightMapEncoderInputNames_.push_back(heightMapEncoderSessionPtr_->GetInputName(i, allocator));
    heightMapEncoderInputShapes_.push_back(heightMapEncoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < heightMapEncoderSessionPtr_->GetOutputCount(); i++) {
    heightMapEncoderOutputNames_.push_back(heightMapEncoderSessionPtr_->GetOutputName(i, allocator));
    heightMapEncoderOutputShapes_.push_back(heightMapEncoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load height map encoder model successfully !!!");

  // policy session
  policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, policyFilePath.c_str(), sessionOptions);
  policyInputNames_.clear();
  policyOutputNames_.clear();
  policyInputShapes_.clear();
  policyOutputShapes_.clear();
  for (size_t i = 0; i < policySessionPtr_->GetInputCount(); i++) {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < policySessionPtr_->GetOutputCount(); i++) {
    policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
    policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load biped vision policy model successfully !!!");

  ROS_INFO_STREAM("------- load linear velocity estimator model -------");
  std::string linVelEstimatorPath;
  if (!nh_.getParam("/linVelEstimatorFile", linVelEstimatorPath)) {
    ROS_ERROR_STREAM(
        "Get linear velocity estimator path fail from param "
        "server, some error occur!");
    return false;
  }
  linVelEstimatorPath_ = linVelEstimatorPath;

  // linear velocity estimator session
  linVelEstimatorInputNames_.clear();
  linVelEstimatorOutputNames_.clear();
  linVelEstimatorInputShapes_.clear();
  linVelEstimatorOutputShapes_.clear();
  linVelEstimatorSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, linVelEstimatorPath_.c_str(), sessionOptions);
  for (size_t i = 0; i < linVelEstimatorSessionPtr_->GetInputCount(); i++) {
    linVelEstimatorInputNames_.push_back(linVelEstimatorSessionPtr_->GetInputName(i, allocator));
    linVelEstimatorInputShapes_.push_back(linVelEstimatorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < linVelEstimatorSessionPtr_->GetOutputCount(); i++) {
    linVelEstimatorOutputNames_.push_back(linVelEstimatorSessionPtr_->GetOutputName(i, allocator));
    linVelEstimatorOutputShapes_.push_back(linVelEstimatorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load linear velocity estimator model successfully !!!");

  ROS_INFO_STREAM("------- load environment parameters estimator model -------");
  std::string envParamsEstimatorPath;
  if (!nh_.getParam("/envParamsEstimatorFile", envParamsEstimatorPath)) {
    ROS_ERROR_STREAM(
        "Get environment parameters estimator path fail from "
        "param server, some error occur!");
    return false;
  }
  envParamsEstimatorPath_ = envParamsEstimatorPath;

  // environment parameters estimator session
  envParamsEstimatorInputNames_.clear();
  envParamsEstimatorOutputNames_.clear();
  envParamsEstimatorInputShapes_.clear();
  envParamsEstimatorOutputShapes_.clear();
  envParamsEstimatorSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, envParamsEstimatorPath_.c_str(), sessionOptions);
  for (size_t i = 0; i < envParamsEstimatorSessionPtr_->GetInputCount(); i++) {
    envParamsEstimatorInputNames_.push_back(envParamsEstimatorSessionPtr_->GetInputName(i, allocator));
    envParamsEstimatorInputShapes_.push_back(envParamsEstimatorSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < envParamsEstimatorSessionPtr_->GetOutputCount(); i++) {
    envParamsEstimatorOutputNames_.push_back(envParamsEstimatorSessionPtr_->GetOutputName(i, allocator));
    envParamsEstimatorOutputShapes_.push_back(envParamsEstimatorSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load environment parameters estimator model successfully !!!");

  ROS_INFO_STREAM("------- load depth backbone model -------");
  std::string depthDepthBackBonePath;
  if (!nh_.getParam("/depthDepthBackBoneFile", depthDepthBackBonePath)) {
    ROS_ERROR_STREAM("Get depth backbone path fail from param server, some error occur!");
    return false;
  }

  // depth backbone session
  depthDepthBackBonePath_ = depthDepthBackBonePath;
  depthDepthBackBoneInputNames_.clear();
  depthDepthBackBoneOutputNames_.clear();
  depthDepthBackBoneInputShapes_.clear();
  depthDepthBackBoneOutputShapes_.clear();
  depthDepthBackBoneSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, depthDepthBackBonePath_.c_str(), sessionOptions);
  for (size_t i = 0; i < depthDepthBackBoneSessionPtr_->GetInputCount(); i++) {
    depthDepthBackBoneInputNames_.push_back(depthDepthBackBoneSessionPtr_->GetInputName(i, allocator));
    depthDepthBackBoneInputShapes_.push_back(depthDepthBackBoneSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < depthDepthBackBoneSessionPtr_->GetOutputCount(); i++) {
    depthDepthBackBoneOutputNames_.push_back(depthDepthBackBoneSessionPtr_->GetOutputName(i, allocator));
    depthDepthBackBoneOutputShapes_.push_back(depthDepthBackBoneSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load depth backbone model successfully !!!");

  ROS_INFO_STREAM("------- load depth belief encoder model -------");
  std::string depthBeliefEncoderPath;
  if (!nh_.getParam("/depthBeliefEncoderFile", depthBeliefEncoderPath)) {
    ROS_ERROR_STREAM(
        "Get depth belief encoder path fail from param server, "
        "some error occur!");
    return false;
  }

  // depth belief encoder session
  depthBeliefEncoderPath_ = depthBeliefEncoderPath;
  depthBeliefEncoderInputNames_.clear();
  depthBeliefEncoderOutputNames_.clear();
  depthBeliefEncoderInputShapes_.clear();
  depthBeliefEncoderOutputShapes_.clear();
  depthBeliefEncoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, depthBeliefEncoderPath_.c_str(), sessionOptions);
  for (size_t i = 0; i < depthBeliefEncoderSessionPtr_->GetInputCount(); i++) {
    depthBeliefEncoderInputNames_.push_back(depthBeliefEncoderSessionPtr_->GetInputName(i, allocator));
    depthBeliefEncoderInputShapes_.push_back(depthBeliefEncoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < depthBeliefEncoderSessionPtr_->GetOutputCount(); i++) {
    depthBeliefEncoderOutputNames_.push_back(depthBeliefEncoderSessionPtr_->GetOutputName(i, allocator));
    depthBeliefEncoderOutputShapes_.push_back(depthBeliefEncoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  ROS_INFO_STREAM("Load belief encoder model successfully !!!");

  return true;
}

bool PointfootVisionController::loadRLCfg() {
  auto& initState = robotCfg_.initState;
  BipedRobotCfg::ControlCfg& controlCfg = robotCfg_.controlCfg;
  BipedRobotCfg::RlCfg::ObsScales& obsScales = robotCfg_.rlCfg.obsScales;

  try {
    int error = 0;
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/joint_names", jointNames_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/init_state/default_joint_angle/abad_L_Joint", initState["abad_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/init_state/default_joint_angle/hip_L_Joint", initState["hip_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/init_state/default_joint_angle/knee_L_Joint", initState["knee_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/init_state/default_joint_angle/abad_R_Joint", initState["abad_R_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/init_state/default_joint_angle/hip_R_Joint", initState["hip_R_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/init_state/default_joint_angle/knee_R_Joint", initState["knee_R_Joint"]));

    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/stand_mode/stand_duration", standDuration_));
    error += static_cast<int>(!nh_.getParam("/robot_hw/loop_frequency", loopFrequency_));

    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/control/stiffness", controlCfg.stiffness));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/control/damping", controlCfg.damping));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/control/action_scale_pos", controlCfg.action_scale_pos));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/control/decimation", controlCfg.decimation));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/control/user_torque_limit", controlCfg.user_torque_limit));

    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/encoder_nomalize", robotCfg_.rlCfg.encoder_nomalize));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/clip_scales/clip_observations", robotCfg_.rlCfg.clipObs));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/clip_scales/clip_actions", robotCfg_.rlCfg.clipActions));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));

    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/actions_size", actionsSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/commands_size", commmandsSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/observations_size", observationsSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/hidden_state_size", hiddenStatesSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/belief_hidden_state_size", beliefHiddenStatesSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/depth_latent_size", depthLatentSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/size/obs_history_length", obsHistoryLen_));

    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/depth_image/original", depthOriginalShape_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/depth_image/resized", depthResizedShape_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/depth_image/near_clip", nearClip_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/depth_image/far_clip", farClip_));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/depth_image/buffer_len", depthBufferLen_));

    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/imu_orientation_offset/yaw", imuOrientationOffset_[0]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/imu_orientation_offset/pitch", imuOrientationOffset_[1]));
    error += static_cast<int>(!nh_.getParam("/PointfootVisionCfg/imu_orientation_offset/roll", imuOrientationOffset_[2]));

    if (error) {
      ROS_ERROR("Load parameters from ROS parameter server error!!!");
    }
    if (standDuration_ <= 0.5) {
      ROS_ERROR("standDuration_ must be larger than 0.5!!!");
    }
    robotCfg_.print();

    // Resize vectors.
    actions_.resize(actionsSize_);
    observations_.resize(observationsSize_);
    lastActions_.resize(actionsSize_);
    hiddenStates_.resize(hiddenStatesSize_);
    beliefHiddenStates_.resize(beliefHiddenStatesSize_);
    depthLatents_.resize(depthLatentSize_);
    beliefLatents_.resize(depthLatentSize_);

    numPixels_ = depthResizedShape_[0] * depthResizedShape_[1];

    lastActions_.setZero();
    commands_.setZero();
    scaled_commands_.setZero();
  } catch (const std::exception& e) {
    // Error handling.
    ROS_ERROR("Error in the PointfootCfg: %s", e.what());
    return false;
  }

  resizedDepthImagePub_ = nh_.advertise<sensor_msgs::Image>("/d435F/aligned_depth_to_color/image_resized", 1, true);
  heightMapPub_ = nh_.advertise<visualization_msgs::Marker>("/height_map", 1, true);
  std::string depthImageTopicName;
  if (!nh_.getParam("/depthImageTopicName", depthImageTopicName)) {
    ROS_ERROR_STREAM("Get depth image topic fail from param server, some error occur!");
    return false;
  }
  depthImageSub_ = nh_.subscribe(depthImageTopicName, 1, &PointfootVisionController::depthImageCallback, this);
  // TODO: here should subscribe the RGB Image

  return true;
}

void PointfootVisionController::depthImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  uint32_t imageWidth = msg->width;
  uint32_t imageHeight = msg->height;
  std::string encoding = msg->encoding;

  assert(imageWidth == depthOriginalShape_[0]);
  assert(imageHeight == depthOriginalShape_[1]);

  std::vector<float> imageData;
  uint32_t imageNumPixel = imageWidth * imageHeight;
  imageData.reserve(imageNumPixel);

  for (size_t i = 0; i < imageNumPixel; i++) {
    uint16_t pixelValue = (msg->data[i * 2 + 1] << 8) | msg->data[i * 2];
    float distance = -static_cast<float>(pixelValue) / 1000;
    distance = std::min(std::max(distance, -farClip_), -nearClip_);
    imageData.push_back(distance);
  }

  std::vector<float> cropImage = cropDepthImage(imageData, imageWidth, imageHeight, 4, 4, 0, 2);

  cv::Mat srcImage(imageHeight - 2, imageWidth - 8, CV_32F, cropImage.data());
  cv::Size targetResize(depthResizedShape_[0], depthResizedShape_[1]);
  cv::Mat resizedImage;
  cv::resize(srcImage, resizedImage, targetResize, 0, 0, cv::INTER_CUBIC);
  std::vector<float> resizedImageData(resizedImage.begin<float>(), resizedImage.end<float>());

  for (size_t i = 0; i < resizedImageData.size(); i++) {
    resizedImageData[i] *= -1;
    resizedImageData[i] = (resizedImageData[i] - nearClip_) / (farClip_ - nearClip_) - 0.5;
  }

  sensor_msgs::Image resizeImgMsg;
  resizeImgMsg.step = msg->step;
  resizeImgMsg.encoding = msg->encoding;
  resizeImgMsg.header.stamp = ros::Time::now();
  resizeImgMsg.header.frame_id = msg->header.frame_id;
  resizeImgMsg.width = depthResizedShape_[0];
  resizeImgMsg.height = depthResizedShape_[1];
  resizeImgMsg.is_bigendian = msg->is_bigendian;
  resizeImgMsg.data.resize(depthResizedShape_[0] * depthResizedShape_[1] * 2);
  for (size_t i = 0; i < resizedImageData.size(); i++) {
    auto distance = static_cast<uint16_t>((resizedImageData[i] + 0.5) * 1000);
    resizeImgMsg.data[i * 2] = distance & 0xFF;
    resizeImgMsg.data[i * 2 + 1] = distance >> 8;
  }
  resizedDepthImagePub_.publish(resizeImgMsg);

  if (isFirstRecDepth_) {
    depthBufferPtr_ = std::make_shared<std::deque<std::vector<float>>>();
    for (size_t i = 0; i < depthBufferLen_; i++) {
      depthBufferPtr_->push_back(resizedImageData);
    }
    isFirstRecDepth_ = false;
  } else {
    if (!depthBufferPtr_->empty()) {
      depthBufferPtr_->pop_front();
      depthBufferPtr_->push_back(resizedImageData);
      latestDepthImageBuffer_.writeFromNonRT(depthBufferPtr_->back());
    } else {
      ROS_ERROR("depth buffer is empty, could not be updated");
    }
  }

  // std::vector<float> visualizeData(depthBufferPtr_->back().begin(),
  // depthBufferPtr_->back().end()); for (size_t i = 0; i <
  // visualizeData.size(); i++) {
  //   visualizeData[i] += 0.5;
  // }
  //
  // // Convert the visualizeData to a cv::Mat for displaying
  // cv::Mat outputImage(depthResizedShape_[1], depthResizedShape_[0], CV_32F,
  // visualizeData.data());
  //
  // // Display the image
  // std::string window_name = "Depth Image";
  // cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  //
  // cv::imshow("Depth Image", outputImage);
  // cv::waitKey(1); // Wait for a key press for 1 ms
}

std::vector<float> PointfootVisionController::cropDepthImage(const std::vector<float>& image, int width, int height, int left, int right,
                                                             int top, int bottom) {
  if (image.empty() || width <= 0 || height <= 0) {
    return {};
  }

  int cropped_width = width - left - right;
  int cropped_height = height - top - bottom;

  if (cropped_width <= 0 || cropped_height <= 0) {
    return {};
  }

  std::vector<float> cropped_image(cropped_width * cropped_height);

  for (size_t i = 0; i < cropped_height; ++i) {
    std::copy(image.begin() + (i + top) * width + left, image.begin() + (i + top) * width + left + cropped_width,
              cropped_image.begin() + i * cropped_width);
  }

  return cropped_image;
}

void PointfootVisionController::visualizeHeightMap(const std::vector<tensor_element_t>& heightMap) {
  const int numPointsX = 16;
  const int numPointsY = 16;

  visualization_msgs::Marker points;
  points.header.frame_id = "odom";
  points.header.stamp = ros::Time::now();
  points.ns = "height_map";
  points.action = visualization_msgs::Marker::ADD;
  points.pose.orientation.w = 1.0;

  points.id = 0;
  points.type = visualization_msgs::Marker::POINTS;

  points.scale.x = 0.2;
  points.scale.y = 0.2;

  points.color.g = 1.0f;
  points.color.a = 1.0;

  for (int i = 0; i < numPointsX; ++i) {
    for (int j = 0; j < numPointsY; ++j) {
      geometry_msgs::Point p;
      p.x = i * 1;
      p.y = j * 1;
      p.z = -heightMap[i * numPointsY + j];
      points.points.push_back(p);
    }
  }
  heightMapPub_.publish(points);
}
}  // namespace robot_controller

// Export the class as a plugin.
PLUGINLIB_EXPORT_CLASS(robot_controller::PointfootVisionController, controller_interface::ControllerBase)
