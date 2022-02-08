import logging.config
import os

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()

def load_config():
    if os.path.exists('./config.ini'):
        try:
            config = {}
            with open('./config.ini', 'rb') as f:
                logger.info('Success to read "config.ini" file.')
                for line in f:
                    line = line.decode('utf-8').strip('\r\n').strip('\n')
                    line = line.split(' = ')
                    if len(line) == 2:
                        logger.debug(f'"{line[0]}" configuration start.')
                        data = line[1].replace(' ', '').split(',')
                        if len(data) > 1:
                            data_list = []
                            for element in data:
                                if "'" in element:
                                    try:
                                        element = element.replace("'", '')
                                        data_list.append(str(element))
                                    except Exception as E:
                                        logger.exception(f'Unknown error occurred while adding data.. ({E})')
                                    else:
                                        logger.debug(f'add "{element}" to data_list as str-type.')
                                else:
                                    try:
                                        data_list.append(int(element))
                                    except ValueError:
                                        logger.warning(f'"{element}" cannot be parsed to an integer type..')
                                        try:
                                            data_list.append(str(element))
                                        except Exception as E:
                                            logger.exception(f'Unknown error occurred while adding data.. ({E})')
                                            exit()
                                        else:
                                            logger.warning(f'add "{element}" to data_list as str-type.')
                                    except Exception as E:
                                        logger.exception(f'Unknown error occurred while adding data.. ({E})')
                            config[line[0]] = data_list
                            logger.debug(f'"{line[0]}" configuration complete.')
                        else:
                            if "'" in data[0]:
                                try:
                                    data[0] = data[0].replace("'", '')
                                    config[line[0]] = str(data[0])
                                except Exception as E:
                                    logger.exception(f'Unknown error occurred while adding data.. ({E})')
                                    exit()
                                else:
                                    logger.debug(f'add "{data[0]}" to config as str-type.')
                                    logger.debug(f'"{line[0]}" configuration complete.')
                            else:
                                try:
                                    config[line[0]] = int(data[0])
                                except TypeError:
                                    logger.warning(f'"{data[0]}" cannot be parsed to an integer type..')
                                    try:
                                        config[line[0]] = str(data)
                                    except Exception as E:
                                        logger.exception(f'Unknown error occurred while adding data.. ({E})')
                                        exit()
                                    else:
                                        logger.warning(f'add "{data}" to data_list as str-type.')
                                        logger.warning(f'"{line[0]}" configuration complete.')
                                except Exception as E:
                                    logger.exception(f'Unknown error occurred while adding data.. ({E})')
                                    exit()
                                else:
                                    logger.debug(f'add "{data[0]}" to config as int-type.')
                                    logger.debug(f'"{line[0]}" configuration complete.')
                    else:
                        pass
            return config
        except Exception as E:
            logger.exception(f'Unknown error occurred.. ({E})')
            exit()
    else:
        logger.error('Cannot found "config.ini" file in current directory..')
        exit()
