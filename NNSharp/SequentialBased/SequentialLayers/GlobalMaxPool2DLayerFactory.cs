using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class GlobalMaxPool2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is GlobalMaxPooling2D)
            {
                GlobalMaxPooling2D pool = descriptor as GlobalMaxPooling2D;

                ILayer layer = new GlobalMaxPool2DLayer();

                return layer;
            }

            return null;
        }
    }
}
