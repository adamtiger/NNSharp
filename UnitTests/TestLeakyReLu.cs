using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using UnitTests.Properties;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UnitTests
{
    [TestClass]
    public class TestLeakyReLu
    {
        [TestMethod]
        public void Test_LeakyReLu_Execute()
        {
            double alpha = 0.3;
            leakyrelu = new LeakyReLuLayer(alpha);

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            leakyrelu.SetInput(data);

            leakyrelu.Execute();

            Data2D output = leakyrelu.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], 4.0, 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], 2.0, 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], alpha*(-2), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], 3.0, 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], alpha * (-1), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], alpha * (-3), 0.00000001);
        }

        /*[TestMethod]
        public void Test_LeakyReLu_KerasModel()
        {
        }*/

        private LeakyReLuLayer leakyrelu;
    }
}
