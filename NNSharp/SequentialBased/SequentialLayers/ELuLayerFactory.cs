using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class ELuLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is ELu) { 

                ELu elu = descriptor as ELu;

                return new ELuLayer(elu.Alpha);
            }

            return null;
        }
    }
}
