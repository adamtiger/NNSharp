using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Reshape2D : IKernelDescriptor
    {
        public Reshape2D(int height, int width, int channel, int batch)
        {
            // The paramters of the new shape.
            this.height = height;
            this.width = width;
            this.channel = channel;
            this.batch = batch;
        }

        public int Height { get { return height; } }
        public int Width { get { return width; } }
        public int Channel { get { return channel; } }
        public int Batch { get { return batch; } }


        private int height;
        private int width;
        private int channel;
        private int batch;
    }
}
