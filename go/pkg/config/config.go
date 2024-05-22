package config

import (
	"os"

	"github.com/pelletier/go-toml/v2"
)

type DBConfig struct {
	URL string `toml:"url"`
}

type Config struct {
	DBConfig DBConfig `toml:"db"`
}

func defaultConfig() *Config {
	return &Config{}
}

func LoadFromFile(path string) (*Config, error) {
	cfg := defaultConfig()

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, err
	}

	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if err := toml.Unmarshal(buf, cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}
