using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using NNSharp.Properties;

namespace NNSharp.BindingCore
{
    static public partial class NativeFunctions
    {
        const string CoreDll = @"D:\ArtificialIntelligence\MachineLearning\git\NNSharp\cppcore\generated\Debug\cppcore.dll";

        [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr create_tensor_integer(int size);

        [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int tensor_integer_get(IntPtr tensor_in, int idx);
    }
}
