libssl=libssl1.0.0_1.0.2n-1ubuntu5_amd64.deb 
libssldev=libssl1.0-dev_1.0.2n-1ubuntu5_amd64.deb
wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl1.0/$libssl
wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl1.0/$libssldev
sudo dpkg  -i $libssl $libssldev 
cd /lib/x86_64-linux-gnu
sudo ln -s libssl.so.1.0.0 libssl.so.10
sudo ln -s libcrypto.so.1.0.0 libcrypto.so.10
