//
// Created by luohx on 24-12-24.
//

#ifndef _LIMX_POINTFOOT_CTS_VISION_CONTROLLER_H_
#define _LIMX_POINTFOOT_CTS_VISION_CONTROLLER_H_

#include "limxsdk/pointfoot.h"
#include "realtime_tools/realtime_buffer.h"
#include "robot_controllers/PointfootController.h"
#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <deque>
#include <future>
#include <mutex>
#include <thread>

namespace robot_controller {

class PointfootCTSVisionController : public PointfootController {
  using tensor_element_t = float;
  using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

 public:
  PointfootCTSVisionController() : futureObj_(exitSignal_.get_future()) {}
  ~PointfootCTSVisionController() override = default;

  void starting(const ros::Time& time) override;
  void stopping(const ros::Time& time) override;

 protected:
  bool loadModel() override;
  bool loadRLCfg() override;
  void computeActions() override;
  void computeObservation() override;
  void computeNormalizedObs(const vector_t& obs);
  void computeDepthBackbone();
  void computeEpEstimate();
  void handleWalkMode() override;

  void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg);
  std::vector<float> cropDepthImage(const std::vector<float>& image, int width, int height, int left, int right, int top, int bottom);
  void visualizeHeightMap(const std::vector<tensor_element_t>& heightMap);

 private:
  // ONNX model paths
  std::string policyPath_;
  std::string normalizerPath_;
  std::string depthBackbonePath_;
  std::string epEstimatorPath_;

  // ONNX runtime environment and sessions
  std::shared_ptr<Ort::Env> onnxEnvPtr_;

  // Policy model
  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::vector<const char*> policyInputNames_;
  std::vector<const char*> policyOutputNames_;
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;

  // Normalizer model
  std::unique_ptr<Ort::Session> normalizerSessionPtr_;
  std::vector<const char*> normalizerInputNames_;
  std::vector<const char*> normalizerOutputNames_;
  std::vector<std::vector<int64_t>> normalizerInputShapes_;
  std::vector<std::vector<int64_t>> normalizerOutputShapes_;

  // Depth backbone model
  std::unique_ptr<Ort::Session> depthBackboneSessionPtr_;
  std::vector<const char*> depthBackboneInputNames_;
  std::vector<const char*> depthBackboneOutputNames_;
  std::vector<std::vector<int64_t>> depthBackboneInputShapes_;
  std::vector<std::vector<int64_t>> depthBackboneOutputShapes_;

  // EP estimator model
  std::unique_ptr<Ort::Session> epEstimatorSessionPtr_;
  std::vector<const char*> epEstimatorInputNames_;
  std::vector<const char*> epEstimatorOutputNames_;
  std::vector<std::vector<int64_t>> epEstimatorInputShapes_;
  std::vector<std::vector<int64_t>> epEstimatorOutputShapes_;

  // State dimensions
  int actionsSize_;
  int commandsSize_;
  int observationsSize_;
  int obsHistoryLen_;
  int epLatentDim_;
  int hmLatentDim_;
  int hiddenStatesSize_;

  // State vectors
  std::vector<tensor_element_t> actions_, observations_, normalizedObs_;
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> observationsHistoryBuffer_;
  std::array<scalar_t, 3> imuOrientationOffset_;
  vector_t lastActions_;

  // Model states
  std::vector<tensor_element_t> hiddenStates_;
  std::vector<tensor_element_t> epLatent_;
  std::vector<tensor_element_t> hmLatent_;
  std::vector<tensor_element_t> predictedHeightMap_;

  // Flags
  bool isFirstRecObs_{true};
  bool isFirstRecDepth_{true};

  // Gait parameters
  double gaitIndex_{0};

  // Depth image parameters
  int numPixels_;
  float farClip_;
  float nearClip_;
  int depthBufferLen_;
  std::vector<int> depthOriginalShape_;
  std::vector<int> depthResizedShape_;
  std::shared_ptr<std::deque<std::vector<tensor_element_t>>> depthBufferPtr_;

  // ROS communication
  ros::Subscriber depthImageSub_;
  ros::Publisher resizedDepthImagePub_;
  ros::Publisher heightMapPub_;

  // Threading
  std::promise<void> exitSignal_;
  std::future<void> futureObj_;
  std::thread computeDepthBackboneThread_;

  // Real-time communication structures
  struct Main2SubParams {
    std::vector<tensor_element_t> normalizedObs;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> normalizedObsHistory;

    Main2SubParams() = default;
    Main2SubParams(const std::vector<tensor_element_t>& obs, const Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1>& obsHistory)
        : normalizedObs(obs), normalizedObsHistory(obsHistory) {}
  };

  struct Sub2MainParams {
    std::vector<tensor_element_t> hmLatent;
    std::vector<tensor_element_t> predictedHeightMap;

    Sub2MainParams() = default;
    Sub2MainParams(const std::vector<tensor_element_t>& hm, const std::vector<tensor_element_t>& phm)
        : hmLatent(hm), predictedHeightMap(phm) {}
  };

  // Real-time buffers
  realtime_tools::RealtimeBuffer<Main2SubParams> main2SubBuffer_;
  realtime_tools::RealtimeBuffer<Sub2MainParams> sub2MainBuffer_;
  realtime_tools::RealtimeBuffer<std::vector<tensor_element_t>> latestDepthImageBuffer_;
};

}  // namespace robot_controller
#endif  // _LIMX_POINTFOOT_CTS_VISION_CONTROLLER_H_
