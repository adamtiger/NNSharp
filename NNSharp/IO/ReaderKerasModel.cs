using Newtonsoft.Json.Linq;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
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
                    case "MaxPooling2D":
                        descriptor = new MaxPooling2D((int)layer.SelectToken("padding_vl"), (int)layer.SelectToken("padding_hz"),
                            (int)layer.SelectToken("stride_vl"), (int)layer.SelectToken("stride_hz"),
                            (int)layer.SelectToken("kernel_height"), (int)layer.SelectToken("kernel_width"));
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
                    case "Input2D":
                        descriptor = new Input2D((int)layer.SelectToken("height"), (int)layer.SelectToken("width"),
                            (int)layer.SelectToken("channel"), (int)layer.SelectToken("batch"));
                        break;
                    case "Bias2D":
                        descriptor = new Bias2D();
                        break;
                    case "ReLu":
                        descriptor = new ReLu();
                        break;
                    case "Flatten":
                        descriptor = new Flatten();
                        break;
                    case "Softmax":
                        descriptor = new Softmax();
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
                if ((dscps[i] is Convolution2D) || (dscps[i] is Dense2D))
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
    }
}
