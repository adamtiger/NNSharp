using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class GlobalAvgPool2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is GlobalAvgPooling2D)
            {
                GlobalAvgPooling2D pool = descriptor as GlobalAvgPooling2D;

                ILayer layer = new GlobalAvgPool2DLayer();

                return layer;
            }

            return null;
        }
    }
}
