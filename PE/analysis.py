class GeneralFileInfo(FeatureType):
     General information about the file ''
  'size': len(bytez),
            'vsize': lief_binary.virtual_size,
	#IMAGE_SECTION_HEADER 메모리에서 섹션이 차지하는 크기 
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
	#IMAGE_EXPORT_DESCRIPTOR 라이브러리 입장에서 다른 PE 파일에게 서비스 제공하는일 
            'imports': len(lief_binary.imported_functions),
	#IMAGE_EXPORT_DESCRIPTOR 라이브러리한테서 서비스를 제공 받는 일 
            'has_relocations': int(lief_binary.has_relocations),
	# PE reloacation 더 조사해봐야할듯 
            'has_resources': int(lief_binary.has_resources),
	#IMAGE_DATA_DIRECTORY 구조체 변수 예상 
            'has_signature': int(lief_binary.has_signature),
	#PE 
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),




class HeaderFileInfo(FeatureType):
    ''' Machine, architecure, OS, linker and other information extracted from header '''


 if lief_binary is None:
            return raw_obj

        raw_obj['coff']['timestamp'] = lief_binary.header.time_date_stamps 
	#IMAGE_FILE_HEADER 오브젝트 생성일자 (GMT시간 기준)
        raw_obj['coff']['machine'] = str(lief_binary.header.machine).split('.')[-1] 
	#IMAGE_FILE_HEADER  CPU별로 고유한 값을 가지면서 IA-32 호환 CPU 14Ch의 값을 IA-64호환 CPU는 200h의 값을 가집니다
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1] for c in lief_binary.header.characteristics_list] 
	# IMAGE_FILE_HEADER 이 값은 파일 속석에 대한 부분 
        raw_obj['optional']['subsystem'] = str(lief_binary.optional_header.subsystem).split('.')[-1] 
	#IMAGE_OPTIONAL_HEADER 32 -1: 드라이버파일(SYS,VXD)이라는 뜻입니다
				    -2: GUI파일이라는 뜻입니다
	                                          -3: CUI파일이라는 뜻입니다
        raw_obj['optional']['dll_characteristics'] = [
            str(c).split('.')[-1] for c in lief_binary.optional_header.dll_characteristics_lists
        ]
        raw_obj['optional']['magic'] = str(lief_binary.optional_header.magic).split('.')[-1]
	#IMAGE_OPTIONAL_HEADER 32이 구조체가 32bit용이면 10Bh 64bit 용이면 20Bh 의 값을 가집니다
        raw_obj['optional']['major_image_version'] = lief_binary.optional_header.major_image_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['minor_image_version'] = lief_binary.optional_header.minor_image_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['major_linker_version'] = lief_binary.optional_header.major_linker_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['minor_linker_version'] = lief_binary.optional_header.minor_linker_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['major_operating_system_version'] = lief_binary.optional_header.major_operating_system_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['minor_operating_system_version'] = lief_binary.optional_header.minor_operating_system_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['major_subsystem_version'] = lief_binary.optional_header.major_subsystem_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['minor_subsystem_version'] = lief_binary.optional_header.minor_subsystem_version
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['sizeof_code'] = lief_binary.optional_header.sizeof_code
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['sizeof_headers'] = lief_binary.optional_header.sizeof_headers
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        raw_obj['optional']['sizeof_heap_commit'] = lief_binary.optional_header.sizeof_heap_commit
	#IMAGE_OPTIONAL_HEADER 32 구조체 변수 예상 
        return raw_obj

 
        ]).astype(np.float32)
