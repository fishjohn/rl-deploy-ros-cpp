//
// Created by suhaokai on 24-10-17.
//

#ifndef _LIMX_POINTFOOT_VISION_CONTROLLER_H_
#define _LIMX_POINTFOOT_VISION_CONTROLLER_H_

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

class PointfootVisionController : public PointfootController {
  using tensor_element_t = float;                                            // Type alias for tensor elements
  using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;  // Type alias for matrices

 public:
  PointfootVisionController() : futureObj_(exitSignal_.get_future()) {}

  ~PointfootVisionController() override = default;

  void starting(const ros::Time& time) override;

  void stopping(const ros::Time& time) override;

 protected:
  // Load the model for the controller
  bool loadModel() override;

  // Load RL configuration settings
  bool loadRLCfg() override;

  // Compute actions for the controller
  void computeActions() override;

  // Compute observations for the controller
  void computeObservation() override;

  // Compute estimated base linear velocity
  void computeLinVelEstimate();

  // Compute env dynamic parameters latent
  void computeEnvParamsEstimate();

  // Compute predicted height map
  void computePredictedHeightMap();

  // Compute height map latent through HMP&MLP
  void computeHeightMapEncoder();

  // Compute depth latent through CNN
  void computeDepthBackBone();

  // Compute height map latent through Belief Encoder
  void computeBeliefEncoder();

  // Handle walk mode
  void handleWalkMode() override;

  void visualizeHeightMap(const std::vector<tensor_element_t>& heightMap);

  // Depth image process
  void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg);
  std::vector<float> cropDepthImage(const std::vector<float>& image, int width, int height, int left, int right, int top, int bottom);

 private:
  std::string policyFilePath_;
  std::shared_ptr<Ort::Env> onnxEnvPtr_;
  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::vector<const char*> policyInputNames_;
  std::vector<const char*> policyOutputNames_;
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;

  std::string depthPredictorPath_;
  std::shared_ptr<Ort::Env> depthPredictorOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> depthPredictorSessionPtr_;
  std::vector<const char*> depthPredictorInputNames_;
  std::vector<const char*> depthPredictorOutputNames_;
  std::vector<std::vector<int64_t>> depthPredictorInputShapes_;
  std::vector<std::vector<int64_t>> depthPredictorOutputShapes_;

  std::string heightMapEncoderPath_;
  std::shared_ptr<Ort::Env> heightMapEncoderOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> heightMapEncoderSessionPtr_;
  std::vector<const char*> heightMapEncoderInputNames_;
  std::vector<const char*> heightMapEncoderOutputNames_;
  std::vector<std::vector<int64_t>> heightMapEncoderInputShapes_;
  std::vector<std::vector<int64_t>> heightMapEncoderOutputShapes_;

  std::string linVelEstimatorPath_;
  std::shared_ptr<Ort::Env> linVelEstimatorOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> linVelEstimatorSessionPtr_;
  std::vector<const char*> linVelEstimatorInputNames_;
  std::vector<const char*> linVelEstimatorOutputNames_;
  std::vector<std::vector<int64_t>> linVelEstimatorInputShapes_;
  std::vector<std::vector<int64_t>> linVelEstimatorOutputShapes_;

  std::string envParamsEstimatorPath_;
  std::shared_ptr<Ort::Env> envParamsEstimatorOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> envParamsEstimatorSessionPtr_;
  std::vector<const char*> envParamsEstimatorInputNames_;
  std::vector<const char*> envParamsEstimatorOutputNames_;
  std::vector<std::vector<int64_t>> envParamsEstimatorInputShapes_;
  std::vector<std::vector<int64_t>> envParamsEstimatorOutputShapes_;

  std::string depthDepthBackBonePath_;
  std::shared_ptr<Ort::Env> depthDepthBackBoneOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> depthDepthBackBoneSessionPtr_;
  std::vector<const char*> depthDepthBackBoneInputNames_;
  std::vector<const char*> depthDepthBackBoneOutputNames_;
  std::vector<std::vector<int64_t>> depthDepthBackBoneInputShapes_;
  std::vector<std::vector<int64_t>> depthDepthBackBoneOutputShapes_;

  std::string depthBeliefEncoderPath_;
  std::shared_ptr<Ort::Env> depthBeliefEncoderOnnxEnvPrt_;
  std::unique_ptr<Ort::Session> depthBeliefEncoderSessionPtr_;
  std::vector<const char*> depthBeliefEncoderInputNames_;
  std::vector<const char*> depthBeliefEncoderOutputNames_;
  std::vector<std::vector<int64_t>> depthBeliefEncoderInputShapes_;
  std::vector<std::vector<int64_t>> depthBeliefEncoderOutputShapes_;

  int actionsSize_;
  int commmandsSize_;
  int observationsSize_;
  int obsHistoryLen_;
  int depthLatentSize_;
  int hiddenStatesSize_;
  int beliefHiddenStatesSize_;

  std::vector<tensor_element_t> actions_, observations_;
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> observationsHistoryBuffer_;
  std::array<scalar_t, 3> imuOrientationOffset_;

  vector_t lastActions_;

  std::vector<tensor_element_t> hiddenStates_;
  std::vector<tensor_element_t> beliefHiddenStates_;
  std::vector<tensor_element_t> depthLatents_;
  std::vector<tensor_element_t> predictedHeightMap_;
  std::vector<tensor_element_t> linVelLatent_;
  std::vector<tensor_element_t> envParamsLatent_;
  std::vector<tensor_element_t> heightMapLatent_;
  std::vector<tensor_element_t> beliefLatents_;

  bool isFirstRecObs_{true};
  bool isFirstRecDepth_{true};

  double gaitIndex_{0};
  // Depth image
  int numPixels_;
  float farClip_;
  float nearClip_;
  int depthBufferLen_;
  std::vector<int> depthOriginalShape_;
  std::vector<int> depthResizedShape_;
  std::shared_ptr<std::deque<std::vector<tensor_element_t>>> depthBufferPtr_;

  ros::Subscriber depthImageSub_;
  ros::Publisher resizedDepthImagePub_;
  ros::Publisher heightMapPub_;

  //// Sub-Thread
  // Thread control
  std::promise<void> exitSignal_;
  std::future<void> futureObj_;
  std::thread computePredictedHeightMapThread_;

  // Data structures
  struct Main2SubParams {
    std::vector<tensor_element_t> linVelLatent;
    std::vector<tensor_element_t> observation;
    vector3_t scaledCommand;

    Main2SubParams() = default;
    Main2SubParams(const std::vector<tensor_element_t>& ll, const std::vector<tensor_element_t>& ot, const vector3_t& sc)
        : linVelLatent(ll), observation(ot), scaledCommand(sc) {}
  };
  struct Sub2MainParams {
    std::vector<tensor_element_t> predictedHeightMap;

    Sub2MainParams() = default;
    Sub2MainParams(const std::vector<tensor_element_t>& m) : predictedHeightMap(m) {}
  };

  // Realtime buffers
  realtime_tools::RealtimeBuffer<Main2SubParams> main2SubBuffer_;
  realtime_tools::RealtimeBuffer<Sub2MainParams> sub2MainBuffer_;
  realtime_tools::RealtimeBuffer<std::vector<tensor_element_t>> latestDepthImageBuffer_;
};
}  // namespace robot_controller
#endif  //_LIMX_POINTFOOT_VISION_CONTROLLER_H_
