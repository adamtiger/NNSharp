using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;

namespace UnitTests
{
    [TestClass]
    public class TestInput2D
    {
        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullInput()
        {
            Data2D data = null;
            Input2DLayer inp = new Input2DLayer();
            inp.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Input2DLayer inp = new Input2DLayer();
            inp.SetInput(data);
        }
    }
}
