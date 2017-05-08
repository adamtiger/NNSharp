using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestTanH
    {
        [TestMethod]
        public void Test_TanH_Execute()
        {
            relu = new TanHLayer();

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            relu.SetInput(data);

            relu.Execute();

            Data2D output = relu.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], tanh(4.0), 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], tanh(2.0), 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], tanh(-2.0), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], tanh(3.0), 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], tanh(-1.0), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], tanh(-3.0), 0.00000001);
        }

        private double tanh(double x)
        {
            return 2.0 / (1 + Math.Exp(-2 * x)) - 1.0;
        }

        private TanHLayer relu;
    }
}
