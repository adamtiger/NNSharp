using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Input2D : IKernelDescriptor
    {
        public Input2D(int height, int width, int channels, int batch)
        {
            this.height = height;
            this.width = width;
            this.channels = channels;
            this.batch = batch;
        } 

        public int Height { get { return height; } }
        public int Width { get { return width; } }
        public int Channels { get { return channels; } }
        public int Batch { get { return batch; } }

        private int height;
        private int width;
        private int channels;
        private int batch;
    }
}
