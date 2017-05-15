using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.DataTypes
{
    public class SequentialModelData
    {

        public SequentialModelData(double executionTime)
        {
            info = new List<LayerData>();
            this.executionTime = executionTime;
        }

        public int GetNumberofLayers()
        {
            return info.Count;
        }

        public LayerData GetLayerDataAt(int idx)
        {
            return info[idx];
        }

        public string GetLayerNameAt(int idx)
        {
            return info[idx].LayerName;
        }

        public void Add(LayerData data)
        {
            info.Add(data);
        }

        public double GetExecutionTime()
        {
            return executionTime;
        }

        public string GetStringRepresentation()
        {
            SequentialModelData info = this;
            StringBuilder builder = new StringBuilder();

            builder.AppendLine("Number of layers: " + info.GetNumberofLayers());
            builder.AppendLine("Measured time of execution (ms): " + info.GetExecutionTime());

            for (int idx = 0; idx < info.GetNumberofLayers(); ++idx)
            {
                builder.AppendLine("Layer Name: " + info.GetLayerNameAt(idx));

                string s = "    Input: height (" + info.GetLayerDataAt(idx).InputHeight +
                    "), width (" + info.GetLayerDataAt(idx).InputWidth +
                    "), depth (" + info.GetLayerDataAt(idx).InputDepth +
                    "), channel (" + info.GetLayerDataAt(idx).InputChannel +
                    "), batch (" + info.GetLayerDataAt(idx).InputBatch + ")";
                builder.AppendLine(s);

                s = "    Output: height (" + info.GetLayerDataAt(idx).OutputHeight +
                    "), width (" + info.GetLayerDataAt(idx).OutputWidth +
                    "), depth (" + info.GetLayerDataAt(idx).OutputDepth +
                    "), channel (" + info.GetLayerDataAt(idx).OutputChannel +
                    "), batch (" + info.GetLayerDataAt(idx).OutputBatch + ")";
                builder.AppendLine(s);
            }

            return builder.ToString();
        }

        private double executionTime;
        private List<LayerData> info;

        public class LayerData
        {
            public LayerData(
                string name, 
                int inputHeight, int inputWidth, int inputDepth, int inputChannel, int inputBatch,
                int outputHeight, int outputWidth, int outputDepth, int outputChannel, int outputBatch)
            {
                LayerName = name;

                InputHeight = inputHeight;
                InputWidth = inputWidth;
                InputDepth = inputDepth;
                InputChannel = inputChannel;
                InputBatch = inputBatch;

                OutputHeight = outputHeight;
                OutputWidth = outputWidth;
                OutputDepth = outputDepth;
                OutputChannel = outputChannel;
                OutputBatch = outputBatch;
            }

            public string LayerName { get; private set; }

            public int InputHeight { get; private set; }
            public int InputWidth { get; private set; }
            public int InputDepth { get; private set; }
            public int InputChannel { get; private set; }
            public int InputBatch { get; private set; }

            public int OutputHeight { get; private set; }
            public int OutputWidth { get; private set; }
            public int OutputDepth { get; private set; }
            public int OutputChannel { get; private set; }
            public int OutputBatch { get; private set; }
        } 
    }
}
