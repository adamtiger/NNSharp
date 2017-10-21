using Newtonsoft.Json.Linq;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
using NNSharp.Kernels;
using NNSharp.Kernels.CPUKernels;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialExecutors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.IO
{
    public class ReaderKerasModel
    {
        public ReaderKerasModel(string fname)
        {
            JObject model = JObject.Parse(File.ReadAllText(fname));
            String modelType = (String)model.SelectToken("model_type");

            if (!modelType.Equals("Sequential"))
                throw new Exception("This reader only supports Sequential type models!");

            SequentialModel seq = new SequentialModel();

            List<IKernelDescriptor> descriptors = ReadDescriptors(model);

            foreach (var d in descriptors)
            {
                seq.Add(d);
            }

            seq.Compile(new DefaultExecutor());

            List<IData> weights = ReadWeights(model, descriptors);

            seq.SetWeights(weights);

            sequential = seq;
        }

        public SequentialModel GetSequentialExecutor()
        {
            return sequential;
        }

        private List<IKernelDescriptor> ReadDescriptors(JObject model)
        {
            List<IKernelDescriptor> dscps = model.SelectToken("descriptors").Select(layer => {

                IKernelDescriptor descriptor = null;

                String layerName = (String)layer.SelectToken("layer");

                switch (layerName)
                {
                    case "AvgPooling1D":
                        descriptor = new AvgPooling1D(
                            (int)layer.SelectToken("padding"),
                            (int)layer.SelectToken("stride"),
                            (int)layer.SelectToken("kernel_size"));
                        break;
                    case "GlobalAveragePooling1D":
                        descriptor = new GlobalAvgPooling1D();
                        break;
                    case "AvgPooling2D":
                        descriptor = new AvgPooling2D((int)layer.SelectToken("padding_vl"), (int)layer.SelectToken("padding_hz"),
                            (int)layer.SelectToken("stride_vl"), (int)layer.SelectToken("stride_hz"),
                            (int)layer.SelectToken("kernel_height"), (int)layer.SelectToken("kernel_width"));
                        break;
                    case "GlobalAveragePooling2D":
                        descriptor = new GlobalAvgPooling2D();
                        break;
                    case "BatchNormalization":
                        descriptor = new BatchNormalization(
                            (int)layer.SelectToken("epsilon"));
                        break;
                    case "Cropping1D":
                        descriptor = new Cropping1D(
                            (int)layer.SelectToken("trimBegin"),
                            (int)layer.SelectToken("trimEnd"));
                        break;
                    case "Cropping2D":
                        descriptor = new Cropping2D(
                            (int)layer.SelectToken("topTrim"),
                            (int)layer.SelectToken("bottomTrim"),
                            (int)layer.SelectToken("leftTrim"),
                            (int)layer.SelectToken("rightTrim"));
                        break;
                    case "MaxPooling1D":
                        descriptor = new MaxPooling1D(
                            (int)layer.SelectToken("padding"),
                            (int)layer.SelectToken("stride"), 
                            (int)layer.SelectToken("kernel_size"));
                        break;
                    case "GlobalMaxPooling1D":
                        descriptor = new GlobalMaxPooling1D();
                        break;
                    case "MaxPooling2D":
                        descriptor = new MaxPooling2D((int)layer.SelectToken("padding_vl"), (int)layer.SelectToken("padding_hz"),
                            (int)layer.SelectToken("stride_vl"), (int)layer.SelectToken("stride_hz"),
                            (int)layer.SelectToken("kernel_height"), (int)layer.SelectToken("kernel_width"));
                        break;
                    case "GlobalMaxPooling2D":
                        descriptor = new GlobalMaxPooling2D();
                        break;
                    case "Convolution1D":
                        descriptor = new Convolution1D(
                            (int)layer.SelectToken("padding"),
                            (int)layer.SelectToken("stride"),
                            (int)layer.SelectToken("kernel_size"),
                            (int)layer.SelectToken("kernel_num"));
                        break;
                    case "Convolution2D":
                        descriptor = new Convolution2D((int)layer.SelectToken("padding_vl"), (int)layer.SelectToken("padding_hz"),
                            (int)layer.SelectToken("stride_vl"), (int)layer.SelectToken("stride_hz"),
                            (int)layer.SelectToken("kernel_height"), (int)layer.SelectToken("kernel_width"),
                            (int)layer.SelectToken("kernel_num"));
                        break;
                    case "Dense2D":
                        descriptor = new Dense2D((int)layer.SelectToken("units"));
                        break;
                    case "Dropout":
                        /*int h = (int)layer.SelectToken("height");
                        int w = (int)layer.SelectToken("weight");
                        int c = (int)layer.SelectToken("channel");
                        int b = (int)layer.SelectToken("batch");*/
                        Data2D noiseShape = new Data2D(1, 2, 3, 2);
                        descriptor = new Dropout((double)layer.SelectToken("rate"), noiseShape);
                        break;
                    case "Input2D":
                        descriptor = new Input2D((int)layer.SelectToken("height"), (int)layer.SelectToken("width"),
                            (int)layer.SelectToken("channel"), (int)layer.SelectToken("batch"));
                        break;
                    case "Bias2D":
                        descriptor = new Bias2D();
                        break;
                    case "Permute":
                        descriptor = new Permute(
                            (int)layer.SelectToken("dim1"),
                            (int)layer.SelectToken("dim2"),
                            (int)layer.SelectToken("dim3"));
                        break;
                    case "Reshape":
                        descriptor = new Reshape2D(
                            (int)layer.SelectToken("height"),
                            (int)layer.SelectToken("width"),
                            (int)layer.SelectToken("channel"),
                            1);
                        break;
                    case "RepeatVector":
                        descriptor = new RepeatVector(
                            (int)layer.SelectToken("num"));
                        break;
                    case "SimpleRNN":
                        descriptor = new SimpleRNN(
                            (int)layer.SelectToken("units"),
                            (int)layer.SelectToken("input_dim"),
                            ANR((string)layer.SelectToken("activation")));
                        break;
                    case "LSTM":
                        descriptor = new LSTM(
                            (int)layer.SelectToken("units"),
                            (int)layer.SelectToken("input_dim"),
                            ANR((string)layer.SelectToken("activation")),
                            ANR((string)layer.SelectToken("rec_act")));
                        break;
                    case "GRU":
                        descriptor = new GRU(
                            (int)layer.SelectToken("units"),
                            (int)layer.SelectToken("input_dim"),
                            ANR((string)layer.SelectToken("activation")),
                            ANR((string)layer.SelectToken("rec_act")));
                        break;
                    case "ELu":
                        descriptor = new ELu(1);
                        break;
                    case "HardSigmoid":
                        descriptor = new HardSigmoid();
                        break;
                    case "ReLu":
                        descriptor = new ReLu();
                        break;
                    case "Sigmoid":
                        descriptor = new Sigmoid();
                        break;
                    case "Flatten":
                        descriptor = new Flatten();
                        break;
                    case "Softmax":
                        descriptor = new Softmax();
                        break;
                    case "SoftPlus":
                        descriptor = new SoftPlus();
                        break;
                    case "SoftSign":
                        descriptor = new Softsign();
                        break;
                    case "TanH":
                        descriptor = new TanH();
                        break;
                    default:
                        throw new Exception("Unknown layer type!");
                }

                return descriptor;
            }).ToList();

            return dscps;
        }

        private List<IData> ReadWeights(JObject model, List<IKernelDescriptor> dscps)
        {
            List<List<List<List<List<double>>>>> weightsList = model.SelectToken("weights").Select(
                    d => d.Select(
                            row => row.Select(
                                    col => col.Select(
                                            channel => channel.Select(
                                                    batch => (double)batch
                                                ).ToList()
                                        ).ToList()
                                ).ToList()
                        ).ToList()
                ).ToList();

            List<IData> weights = new List<IData>();

            int idx = 0;
            for (int i = 1; i < dscps.Count; ++i)
            {
                if ((dscps[i] is Convolution2D) || (dscps[i] is Dense2D) || (dscps[i] is Convolution1D) || 
                    dscps[i] is BatchNormalization || dscps[i] is SimpleRNN || dscps[i] is LSTM ||
                    dscps[i] is GRU)
                {
                    int rowNum = weightsList[idx].Count;
                    int colNum = weightsList[idx][0].Count;
                    int chNum = weightsList[idx][0][0].Count;
                    int batchNum = weightsList[idx][0][0][0].Count;

                    Data2D data = new Data2D(rowNum, colNum, chNum, batchNum);

                    for (int row = 0; row < rowNum; ++row)
                    {
                        for (int col = 0; col < colNum; ++col)
                        {
                            for (int chnl = 0; chnl < chNum; ++chnl)
                            {
                                for (int batch = 0; batch < batchNum; ++batch)
                                {
                                    data[row, col, chnl, batch] = weightsList[idx][row][col][chnl][batch];
                                }
                            }
                        }
                    }
                    idx += 1;
                    weights.Add(data);
                }
                else if (dscps[i] is Bias2D)
                {
                    int batchNum = weightsList[idx][0][0][0].Count;

                    DataArray data = new DataArray(batchNum);


                    for (int batch = 0; batch < batchNum; ++batch)
                    {
                        data[batch] = weightsList[idx][0][0][0][batch];
                    }

                    idx += 1;
                    weights.Add(data);
                }
                else
                {
                    weights.Add(null);
                }
            }

            return weights;
        }

        private SequentialModel sequential;

        // Activation Name Resolver
        private ActivationLambda ANR(string name)
        {
            switch (name)
            {
                case "linear":
                    return (x => { });
                case "softmax":
                    return SoftmaxKernel.SoftmaxLambda;
                case "tanh":
                    return TanHKernel.TanHLambda;
                case "elu":
                    return ELuKernel.ELuLambda; // alpha = 1.0
                case "softplus":
                    return SoftPlusKernel.SoftPlusLambda;
                case "softsign":
                    return SoftsignKernel.SoftsignLambda;
                case "relu":
                    return ReLuKernel.ReLuLambda;
                case "sigmoid":
                    return SigmoidKernel.SigmoidLambda;
                case "hard_sigmoid":
                    return HardSigmoidKernel.HardSigmoidLambda;
                default:
                    throw new Exception("Unknown activation.");
            }
        }
    }
}
